#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import (
    AlbertModel,
    BertTokenizer,
)


class BinaryVideoWatchPredictor(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_videos: int,
        num_user_features: int,
        num_video_features: int,
        tokenizer: BertTokenizer,
        pos_weight: float,
        embedding_dim: int = 128,
        albert_path: str | None = None,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim

        self.user_id_emb = nn.Embedding(num_users, embedding_dim)
        self.video_id_emb = nn.Embedding(num_videos, embedding_dim)
        self.user_feat_emb = nn.Embedding(
            num_user_features, embedding_dim, padding_idx=0
        )
        self.video_feat_emb = nn.Embedding(
            num_video_features, embedding_dim, padding_idx=0
        )

        self.bert = AlbertModel.from_pretrained(albert_path, output_hidden_states=True)
        self.bert.eval()
        for p in self.bert.parameters():
            p.requires_grad = False

        self.fc_text = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        self.hot_proj = nn.Sequential(
            nn.Linear(4, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        self.fc_final = nn.Sequential(
            nn.Linear(embedding_dim * 6, 128),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 1),
        )

        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float))

    @torch.no_grad()
    def encode_text(self, texts, device):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        outputs = self.bert(**inputs, return_dict=True)
        cls_stack = torch.stack(outputs.hidden_states[-4:], dim=0)  # (4,B,L,H)
        cls_vec = cls_stack.mean(0)[:, 0]  # (B,H)
        return self.fc_text(cls_vec)  # (B,D)

    @staticmethod
    def masked_avg(emb, mask):
        """
        emb  : (B, L, D)
        mask : (B, L, 1)   True 表示有效
        """
        s = (emb * mask).sum(1)  # (B,D)
        denom = mask.sum(1).clamp_min(1e-5)  # 防 0
        return s / denom

    def forward(
        self,
        user_id,
        user_feat,
        video_id,
        video_feat,
        hot_feat,
        text,
        labels=None,
    ):
        device = user_id.device

        u_id_emb = self.user_id_emb(user_id)  # (B,D)
        v_id_emb = self.video_id_emb(video_id)  # (B,D)

        u_emb_all = self.user_feat_emb(user_feat)  # (B,Lu,D)
        v_emb_all = self.video_feat_emb(video_feat)  # (B,Lv,D)
        u_mask = (user_feat != 0).unsqueeze(-1)
        v_mask = (video_feat != 0).unsqueeze(-1)
        u_feat_emb = self.masked_avg(u_emb_all, u_mask)  # (B,D)
        v_feat_emb = self.masked_avg(v_emb_all, v_mask)  # (B,D)

        text_emb = self.encode_text(text, device)  # (B,D)
        hot_emb = self.hot_proj(hot_feat.to(device))  # (B,D)

        x = torch.cat(
            [u_id_emb, u_feat_emb, v_id_emb, v_feat_emb, text_emb, hot_emb],
            dim=-1,  # (B, 6*D)
        )
        logits = self.fc_final(x).squeeze(-1)  # (B,)

        if labels is not None:
            labels = labels.float().to(device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


class BinaryVideoWatchPredictorAttn(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_videos: int,
        num_user_features: int,
        num_video_features: int,
        tokenizer: BertTokenizer,
        pos_weight: float,
        embedding_dim: int = 128,
        albert_path: str | None = None,
        n_transformer_layers: int = 3,
        n_heads: int = 8,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim

        self.user_id_emb = nn.Embedding(num_users, embedding_dim)
        self.video_id_emb = nn.Embedding(num_videos, embedding_dim)
        self.user_feat_emb = nn.Embedding(
            num_user_features, embedding_dim, padding_idx=0
        )
        self.video_feat_emb = nn.Embedding(
            num_video_features, embedding_dim, padding_idx=0
        )

        self.bert = AlbertModel.from_pretrained(albert_path, output_hidden_states=True)
        self.bert.eval()
        for p in self.bert.parameters():
            p.requires_grad = False
        self.fc_text = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        self.hot_proj = nn.Sequential(  # 4 → D
            nn.Linear(4, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout_p,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
            norm=nn.LayerNorm(embedding_dim),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_emb = nn.Parameter(torch.zeros(1, 7, embedding_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(embedding_dim // 2, 1),
        )

        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float))

    @torch.no_grad()
    def encode_text(self, texts, device):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        outputs = self.bert(**inputs, return_dict=True)
        cls_stack = torch.stack(outputs.hidden_states[-4:], dim=0)  # (4,B,L,H)
        cls_vec = cls_stack.mean(0)[:, 0]  # (B,H)
        return self.fc_text(cls_vec)  # (B,D)

    @staticmethod
    def masked_avg(emb, mask):
        s = (emb * mask).sum(1)
        denom = mask.sum(1).clamp_min(1e-5)
        return s / denom

    def forward(
        self,
        user_id,
        user_feat,
        video_id,
        video_feat,
        hot_feat,
        text,
        labels=None,
    ):
        device = user_id.device

        u_id_emb = self.user_id_emb(user_id)  # (B,D)
        v_id_emb = self.video_id_emb(video_id)  # (B,D)

        u_emb_all = self.user_feat_emb(user_feat)  # (B,Lu,D)
        v_emb_all = self.video_feat_emb(video_feat)  # (B,Lv,D)
        u_mask = (user_feat != 0).unsqueeze(-1)
        v_mask = (video_feat != 0).unsqueeze(-1)
        u_feat_emb = self.masked_avg(u_emb_all, u_mask)  # (B,D)
        v_feat_emb = self.masked_avg(v_emb_all, v_mask)  # (B,D)

        text_emb = self.encode_text(text, device)  # (B,D)
        hot_emb = self.hot_proj(hot_feat.to(device))  # (B,D)

        B = user_id.size(0)
        cls_tok = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        token_seq = torch.stack(
            [u_id_emb, u_feat_emb, v_id_emb, v_feat_emb, text_emb, hot_emb],
            dim=1,  # (B,6,D)
        )
        seq = torch.cat([cls_tok, token_seq], dim=1)  # (B,7,D)
        seq = seq + self.pos_emb  # 位置编码

        hidden = self.transformer(seq)  # (B,7,D)
        cls_hidden = hidden[:, 0]  # (B,D)

        logits = self.classifier(cls_hidden).squeeze(-1)  # (B,)

        if labels is not None:
            labels = labels.float().to(device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}
