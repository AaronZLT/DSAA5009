#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import pathlib
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from transformers import (
    BertTokenizer,
    Trainer,
    TrainingArguments,
    AlbertModel,
)
from safetensors.torch import load_file as safe_load

from model import BinaryVideoWatchPredictor, BinaryVideoWatchPredictorFMAttn
from data import (
    load_jsonl_dataset,
    collate_fn,
    compute_metrics,
)
from eval import predict, eval_acc_pre_recall_f1


def scan_best_threshold(probs, labels, steps=200, focus_metric="f1"):
    best_t, best_val, best_stat = 0.5, -1, None
    for t in np.linspace(1 / steps, 1 - 1 / steps, steps - 1):
        preds = (probs >= t).astype(int)
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        cur = dict(accuracy=acc, precision=p, recall=r, f1=f1)[focus_metric]
        if cur > best_val:
            best_t, best_val = t, cur
            best_stat = dict(accuracy=acc, precision=p, recall=r, f1=f1)
    return best_t, best_stat


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        required=True,
    )
    ap.add_argument("--dataset", required=True, help="推理用 .jsonl 数据集")
    ap.add_argument("--output_dir", default="./scan_out")
    ap.add_argument("--base_model", default="uer/albert-base-chinese-cluecorpussmall")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument(
        "--metric", choices=["accuracy", "precision", "recall", "f1"], default="f1"
    )
    return ap.parse_args()


def load_weights(model, ckpt_path):
    ckpt = pathlib.Path(ckpt_path)
    if ckpt.suffix == ".safetensors":
        state = safe_load(str(ckpt))
    else:  # .bin
        state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state, strict=True)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    test_ds = load_jsonl_dataset(args.dataset)
    # test_ds = test_ds.select(range(1024))
    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    model = BinaryVideoWatchPredictorFMAttn(
        num_users=7177,
        num_videos=10729,
        num_user_features=1076,
        num_video_features=31,
        tokenizer=tokenizer,
        albert_path=args.base_model,
        embedding_dim=64,
        pos_weight=12.4,
    )
    load_weights(model, args.checkpoint)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp_eval",
            per_device_eval_batch_size=512,
            dataloader_drop_last=False,
            fp16=False,
        ),
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )

    labels, preds, probs = predict(model, trainer, test_ds, probs_threshold=0.5)
    print("\n=== Metrics @ threshold 0.5 ===")
    eval_acc_pre_recall_f1(labels, preds, os.path.join(args.output_dir, "th_0.5"))

    best_t, best_stat = scan_best_threshold(probs, labels, args.steps, args.metric)
    print(f"\n[Best threshold={best_t:.3f}  based on {args.metric.upper()}]")

    _, best_preds, _ = predict(model, trainer, test_ds, probs_threshold=best_t)
    eval_acc_pre_recall_f1(labels, best_preds, os.path.join(args.output_dir, "best_th"))

    np.save(os.path.join(args.output_dir, "probs.npy"), probs)
    np.save(os.path.join(args.output_dir, "labels.npy"), labels)
    scan_res = dict(
        best_threshold=float(best_t),
        **best_stat,
        roc_auc=float(roc_auc_score(labels, probs)),
        pr_auc=float(average_precision_score(labels, probs)),
    )
    with open(
        os.path.join(args.output_dir, "scan_metrics.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(scan_res, f, ensure_ascii=False, indent=2)
    print(json.dumps(scan_res, indent=2, ensure_ascii=False))
    print(f"\nAll outputs saved under {args.output_dir}")


if __name__ == "__main__":
    main()
