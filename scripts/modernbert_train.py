"""
Experiment A: Fine-tune ModernBERT-base as a sequence classifier.

Run (v1 original data):
  /Users/aditya/venvs/pytorch_practice/bin/python scripts/modernbert_train.py --data v1

Run (v2 augmented data):
  /Users/aditya/venvs/pytorch_practice/bin/python scripts/modernbert_train.py --data v2
"""

import argparse
import json
import os
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset

LABELS = [
    "billing",
    "account_access",
    "refund",
    "product_how_to",
    "bug_report",
    "cancellation",
    "delivery",
    "other",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

MODEL_NAME = "answerdotai/ModernBERT-base"
DATA_FILES = {
    "v1": ("data/train_original.jsonl", "data/val.jsonl"),
    "v2": ("data/train.jsonl", "data/val.jsonl"),
}


def load_jsonl(path):
    examples = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            # Extract text after "Message: " in the user turn
            user_content = record["messages"][0]["content"]
            text = user_content.split("Message: ", 1)[1].strip()
            label = record["messages"][1]["content"].strip()
            examples.append((text, LABEL2ID[label]))
    return examples


class IntentDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.encodings = tokenizer(
            [t for t, _ in examples],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor([l for _, l in examples])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "accuracy": (preds == labels).mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()

    train_path, val_path = DATA_FILES[args.data]
    out_dir = f"models/modernbert/{args.data}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device} | Data: {args.data}")

    train_examples = load_jsonl(train_path)
    val_examples = load_jsonl(val_path)
    print(f"Train: {len(train_examples)} | Val: {len(val_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = IntentDataset(train_examples, tokenizer)
    val_dataset = IntentDataset(val_examples, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        use_mps_device=(device == "mps"),
        report_to="none",
        logging_steps=20,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    metrics = trainer.evaluate()
    print(f"\nFinal val accuracy: {metrics['eval_accuracy']:.3f}")
    print(f"Final val macro-F1: {metrics['eval_macro_f1']:.3f}")

    tokenizer.save_pretrained(out_dir)
    trainer.save_model(out_dir)
    print(f"\nModel saved to {out_dir}")


if __name__ == "__main__":
    main()
