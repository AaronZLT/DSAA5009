#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import (
    AlbertModel,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)


class SelfAttnPool(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # (B,L,1) → (B,L)
        score = self.score(emb).squeeze(-1)
        score = score.masked_fill(~mask, -1e9)
        weight = torch.softmax(score, dim=-1)  # (B,L)
        pooled = (emb * weight.unsqueeze(-1)).sum(dim=1)
        return pooled


class FMInteraction(nn.Module):
    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        sum_emb = emb.sum(dim=1)  # (B,D)
        sum_square = sum_emb.pow(2)  # (B,D)
        square_sum = (emb.pow(2)).sum(dim=1)
        fm = 0.5 * (sum_square - square_sum).sum(dim=1, keepdim=True)  # (B,1)
        return fm


class BinaryVideoWatchPredictorFM(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_videos: int,
        num_user_features: int,
        num_video_features: int,
        tokenizer: BertTokenizer,
        pos_weight: float,
        embedding_dim: int = 64,
        albert_path: str = None,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        self.user_id_emb = nn.Embedding(num_users, embedding_dim)
        self.video_id_emb = nn.Embedding(num_videos, embedding_dim)
        self.user_feat_emb = nn.Embedding(
            num_user_features, embedding_dim, padding_idx=0
        )
        self.video_feat_emb = nn.Embedding(
            num_video_features, embedding_dim, padding_idx=0
        )

        self.bert = AlbertModel.from_pretrained(albert_path, output_hidden_states=True)
        self.fc_text = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        self.fc_deep = nn.Sequential(
            nn.Linear(embedding_dim * 5, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 1),  # (B,1)
        )

        self.fm = FMInteraction()  # 无可学习参数

        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float))

    def encode_text(self, texts, device):
        toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        outs = self.bert(**toks, return_dict=True)
        cls = torch.stack(outs.hidden_states[-4:], 0).mean(0)[:, 0]
        return self.fc_text(cls)  # (B,D)

    def masked_avg(self, emb, mask):
        s = (emb * mask).sum(1)
        return s / mask.sum(1).clamp_min(1e-5)

    def forward(
        self,
        user_id,
        user_feat,
        video_id,
        video_feat,
        text,
        labels=None,
    ):
        device = user_id.device

        u_id = self.user_id_emb(user_id)  # (B,D)
        v_id = self.video_id_emb(video_id)  # (B,D)

        u_feat_mean = self.masked_avg(
            self.user_feat_emb(user_feat), (user_feat != 0).unsqueeze(-1)
        )  # (B,D)
        v_feat_mean = self.masked_avg(
            self.video_feat_emb(video_feat), (video_feat != 0).unsqueeze(-1)
        )  # (B,D)

        txt = self.encode_text(text, device)  # (B,D)

        fields = torch.stack(
            [u_id, u_feat_mean, v_id, v_feat_mean, txt], dim=1
        )  # (B,5,D)
        fm_logit = self.fm(fields)  # (B,1)

        deep_in = torch.cat([u_id, u_feat_mean, v_id, v_feat_mean, txt], dim=-1)
        deep_logit = self.fc_deep(deep_in)  # (B,1)

        logits = (fm_logit + deep_logit).squeeze(-1)  # (B,)

        if labels is not None:
            labels = labels.to(device).float()
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=self.pos_weight
            )
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


class BinaryVideoWatchPredictorFMAttn(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_videos: int,
        num_user_features: int,
        num_video_features: int,
        tokenizer: BertTokenizer,
        pos_weight: float,
        embedding_dim: int = 64,
        albert_path: str | None = None,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        self.user_id_emb = nn.Embedding(num_users, embedding_dim)
        self.video_id_emb = nn.Embedding(num_videos, embedding_dim)
        self.user_feat_emb = nn.Embedding(
            num_user_features, embedding_dim, padding_idx=0
        )
        self.video_feat_emb = nn.Embedding(
            num_video_features, embedding_dim, padding_idx=0
        )

        self.user_pool = SelfAttnPool(embedding_dim)
        self.video_pool = SelfAttnPool(embedding_dim)

        self.bert = AlbertModel.from_pretrained(albert_path, output_hidden_states=True)
        self.fc_text = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        self.fc_deep = nn.Sequential(
            nn.Linear(embedding_dim * 5, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 1),
        )

        self.fm = FMInteraction()

        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float))

    def encode_text(self, texts, device):
        toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        outs = self.bert(**toks, return_dict=True)
        cls = torch.stack(outs.hidden_states[-4:], 0).mean(0)[:, 0]
        return self.fc_text(cls)  # (B,D)

    def forward(
        self,
        user_id,
        user_feat,
        video_id,
        video_feat,
        text,
        labels=None,
    ):
        B, device = user_id.size(0), user_id.device

        u_id = self.user_id_emb(user_id)  # (B,D)
        v_id = self.video_id_emb(video_id)  # (B,D)

        u_seq = self.user_feat_emb(user_feat)  # (B,Lu,D)
        v_seq = self.video_feat_emb(video_feat)  # (B,Lv,D)

        u_mask = user_feat != 0  # (B,Lu)  True=valid
        v_mask = video_feat != 0  # (B,Lv)

        u_feat = self.user_pool(u_seq, u_mask)  # (B,D)
        v_feat = self.video_pool(v_seq, v_mask)  # (B,D)

        txt = self.encode_text(text, device)  # (B,D)

        fields = torch.stack([u_id, u_feat, v_id, v_feat, txt], dim=1)  # (B,5,D)
        fm_logit = self.fm(fields)  # (B,1)

        deep_in = torch.cat([u_id, u_feat, v_id, v_feat, txt], dim=-1)
        deep_logit = self.fc_deep(deep_in)  # (B,1)

        logits = (fm_logit + deep_logit).squeeze(-1)  # (B,)

        if labels is not None:
            labels = labels.to(device).float()
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=self.pos_weight
            )
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


class BinaryVideoWatchPredictor(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_videos: int,
        num_user_features: int,
        num_video_features: int,
        tokenizer: BertTokenizer,
        pos_weight: float,
        embedding_dim: int = 64,
        albert_path: str = None,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        self.user_id_emb = nn.Embedding(num_users, embedding_dim)
        self.video_id_emb = nn.Embedding(num_videos, embedding_dim)
        self.user_feat_emb = nn.Embedding(
            num_user_features, embedding_dim, padding_idx=0
        )
        self.video_feat_emb = nn.Embedding(
            num_video_features, embedding_dim, padding_idx=0
        )

        self.bert = AlbertModel.from_pretrained(
            albert_path,
            output_hidden_states=True,
        )

        self.fc_text = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        self.fc_final = nn.Sequential(
            nn.Linear(embedding_dim * 5, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 1),
        )

        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float))

    def encode_text(self, texts, device):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.bert(**inputs, return_dict=True)
        cls_stack = torch.stack(outputs.hidden_states[-4:], dim=0)  # (4, B, L, H)
        cls_vec = cls_stack.mean(0)[:, 0]  # (B, H)
        return self.fc_text(cls_vec)  # (B, D)

    def masked_avg(self, emb, mask):
        """emb: (B, L, D)  mask: (B, L, 1)"""
        s = (emb * mask).sum(1)  # (B, D)
        denom = mask.sum(1).clamp_min(1e-5)  # 防 0
        return s / denom

    def forward(
        self,
        user_id,
        user_feat,
        video_id,
        video_feat,
        text,
        labels=None,
    ):
        device = user_id.device

        u_id_emb = self.user_id_emb(user_id)  # (B, D)
        v_id_emb = self.video_id_emb(video_id)  # (B, D)

        u_emb_all = self.user_feat_emb(user_feat)  # (B, Lu, D)
        v_emb_all = self.video_feat_emb(video_feat)  # (B, Lv, D)
        u_mask = (user_feat != 0).unsqueeze(-1)  # padding=0
        v_mask = (video_feat != 0).unsqueeze(-1)
        u_feat_emb = self.masked_avg(u_emb_all, u_mask)
        v_feat_emb = self.masked_avg(v_emb_all, v_mask)

        text_emb = self.encode_text(text, device)  # (B, D)

        x = torch.cat(
            [u_id_emb, u_feat_emb, v_id_emb, v_feat_emb, text_emb],
            dim=-1,
        )
        logits = self.fc_final(x).squeeze(-1)  # (B,)

        if labels is not None:
            labels = labels.to(device).float()
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}
