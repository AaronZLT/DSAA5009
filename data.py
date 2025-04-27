#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def preprocess(ex):
    hot_feat = np.array(
        [
            float(ex["play_progress"]),
            np.log1p(ex["like_cnt"]),
            np.log1p(ex["comment_cnt"]),
            np.log1p(ex["follow_cnt"]),
        ],
        dtype=np.float32,
    )

    return {
        "user_id": ex["user_id"],
        "user_feat": ex["user_feat"],
        "video_id": ex["video_id"],
        "video_feat": ex["video_feat"],
        "text": f"{ex['caption']} [SEP] {' '.join(ex['tags'])}",
        "labels": int(ex["labels"]),
        "hot_feat": hot_feat,
    }


def load_jsonl_dataset(path):
    ds = load_dataset("json", data_files=path, split="train")
    return ds.map(preprocess, num_proc=os.cpu_count())


def collate_fn(batch):
    user_id = torch.tensor([b["user_id"] for b in batch], dtype=torch.long)
    user_feat = torch.tensor([b["user_feat"] for b in batch], dtype=torch.long)
    video_id = torch.tensor([b["video_id"] for b in batch], dtype=torch.long)

    max_len = max(len(b["video_feat"]) for b in batch)
    video_feat = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        l = len(b["video_feat"])
        video_feat[i, :l] = torch.tensor(b["video_feat"], dtype=torch.long)

    hot_feat = torch.stack([torch.tensor(b["hot_feat"]) for b in batch])  # (B,4)

    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.float)

    return {
        "user_id": user_id,
        "user_feat": user_feat,
        "video_id": video_id,
        "video_feat": video_feat,
        "hot_feat": hot_feat,
        "text": texts,
        "labels": labels,
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()
    labels = labels.astype(int)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}
