import os
import json

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def predict(model, trainer, test_ds, probs_threshold=0.5):
    model.eval()
    test_pred = trainer.predict(test_ds)
    logits = test_pred.predictions
    labels = test_pred.label_ids  # (N,)
    probs = torch.sigmoid(torch.tensor(logits)).numpy().reshape(-1)  # (N,)
    preds = (probs >= probs_threshold).astype(int)
    return labels, preds, probs


def eval_acc_pre_recall_f1(labels, preds, output_dir):
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    os.makedirs(output_dir, exist_ok=True)
    metrics = {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Metrics saved to {metrics_path}")
    return metrics
