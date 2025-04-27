#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AlbertModel,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

from model import BinaryVideoWatchPredictor, BinaryVideoWatchPredictorFMAttn
from data import (
    load_jsonl_dataset,
    collate_fn,
    compute_metrics,
)
from eval import predict, eval_acc_pre_recall_f1


import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE with logits gives -log(pₜ)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p, 1 - p)  # pₜ
        alpha_t = torch.where(
            targets == 1,  # αₜ
            self.alpha,
            1 - self.alpha,
        )

        focal_term = (1 - p_t) ** self.gamma
        loss = alpha_t * focal_term * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalTrainer(Trainer):
    def __init__(self, alpha=0.25, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha, gamma)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs["logits"]
        loss = self.focal_loss(logits, labels.to(logits.device))

        if return_outputs:
            return loss, outputs
        return loss


def parse_args():
    parser = argparse.ArgumentParser(description="Binary video watch predictor")

    parser.add_argument(
        "--base_model", type=str, default="uer/albert-base-chinese-cluecorpussmall"
    )
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)

    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()

    train_ds = load_jsonl_dataset(args.train_dataset)
    eval_ds = load_jsonl_dataset(args.eval_dataset)
    test_ds = load_jsonl_dataset(args.test_dataset)

    # train_ds = train_ds.select(range(16384))
    # eval_ds = eval_ds.select(range(16384))
    # test_ds = test_ds.select(range(16384))

    tokenizer = BertTokenizer.from_pretrained(args.base_model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy="epoch",
        remove_unused_columns=False,
        bf16=True,
        tf32=True,
        optim="adamw_torch",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )
    """
    BinaryVideoWatchPredictorFMAttn with FocalTrainer
    """
    # model = BinaryVideoWatchPredictorFMAttn(
    #     num_users=7177,
    #     num_videos=10729,
    #     num_user_features=1076,
    #     num_video_features=31,
    #     tokenizer=tokenizer,
    #     albert_path=args.base_model,
    #     embedding_dim=64,
    #     pos_weight=12.4,
    # )
    # trainer = FocalTrainer(
    #     model=model,
    #     alpha=0.9,
    #     gamma=2.0,
    #     args=training_args,
    #     tokenizer=tokenizer,
    #     train_dataset=train_ds,
    #     eval_dataset=eval_ds,
    #     data_collator=collate_fn,
    #     compute_metrics=compute_metrics,
    # )

    """
    BinaryVideoWatchPredictor with original Trainer
    """
    model = BinaryVideoWatchPredictor(
        num_users=7177,
        num_videos=10729,
        num_user_features=1076,
        num_video_features=31,
        tokenizer=tokenizer,
        albert_path=args.base_model,
        embedding_dim=64,
        pos_weight=12.4,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    labels, preds, probs = predict(model, trainer, test_ds, probs_threshold=0.653)
    metrics = eval_acc_pre_recall_f1(labels, preds, args.output_dir)


if __name__ == "__main__":
    main()
