"""
Experiment B: Train a SetFit classifier (contrastive sentence-transformer).

Two variants per data version:
  --mode full     — use entire train split
  --mode fewshot  — 16 examples per class (tests sample-efficiency)

Run:
  /Users/aditya/venvs/pytorch_practice/bin/python scripts/setfit_train.py --data v1 --mode full
  /Users/aditya/venvs/pytorch_practice/bin/python scripts/setfit_train.py --data v1 --mode fewshot
  /Users/aditya/venvs/pytorch_practice/bin/python scripts/setfit_train.py --data v2 --mode full
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

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

BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"
DATA_FILES = {
    "v1": "data/train_original.jsonl",
    "v2": "data/train.jsonl",
}
FEWSHOT_PER_CLASS = 16
SEED = 42


def load_jsonl(path):
    examples = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            user_content = record["messages"][0]["content"]
            text = user_content.split("Message: ", 1)[1].strip()
            label = record["messages"][1]["content"].strip()
            examples.append({"text": text, "label": LABEL2ID[label]})
    return examples


def sample_fewshot(examples, n_per_class, seed):
    random.seed(seed)
    by_class = defaultdict(list)
    for ex in examples:
        by_class[ex["label"]].append(ex)
    sampled = []
    for label_id in range(len(LABELS)):
        pool = by_class[label_id]
        sampled.extend(random.sample(pool, min(n_per_class, len(pool))))
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["v1", "v2"], default="v1")
    parser.add_argument("--mode", choices=["full", "fewshot"], default="full")
    args = parser.parse_args()

    out_dir = f"models/setfit/{args.data}_{args.mode}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Backbone: {BACKBONE} | Data: {args.data} | Mode: {args.mode}")

    examples = load_jsonl(DATA_FILES[args.data])

    if args.mode == "fewshot":
        examples = sample_fewshot(examples, FEWSHOT_PER_CLASS, SEED)
        print(f"Few-shot sample: {len(examples)} examples ({FEWSHOT_PER_CLASS}/class)")
    else:
        print(f"Full train set: {len(examples)} examples")

    train_dataset = Dataset.from_list(examples)

    model = SetFitModel.from_pretrained(
        BACKBONE,
        labels=LABELS,
    )

    training_args = TrainingArguments(
        batch_size=16,
        num_epochs=1,
        num_iterations=20,
        output_dir=out_dir,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained(out_dir)
    print(f"\nModel saved to {out_dir}")


if __name__ == "__main__":
    main()
