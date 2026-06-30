"""
Unified eval harness. Runs synthetic (val set) + natural (10 hand-written messages)
eval on ModernBERT and SetFit models, then prints a comparison table.

Usage:
  /Users/aditya/venvs/pytorch_practice/bin/python scripts/eval_compare.py

Qwen baseline numbers are pulled from the README and included in the table as
pre-filled rows — re-running Qwen inference requires mlx-lm and is not done here.

NOTE on comparability: Qwen receives an explicit label list in its inference prompt
("Classify into one of: [billing, ...]"). ModernBERT and SetFit are discriminative
classifiers that never see the label list at inference time — only the raw message.
This is structurally correct for each model type, not an oversight.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from setfit import SetFitModel

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

# Exact pairs from natural_test.py — do not modify
NATURAL_TESTS = [
    ("billing",         "I just got my monthly statement and the numbers don't add up."),
    ("account_access",  "Locked out again — reset my password twice and it still won't let me in."),
    ("refund",          "I sent the item back two weeks ago and haven't heard anything about my money."),
    ("product_how_to",  "How do I actually use the bulk export feature? I can't figure it out."),
    ("bug_report",      "Every time I hit submit the whole page just goes white and nothing happens."),
    ("cancellation",    "I don't want to keep paying for this. How do I stop my subscription?"),
    ("delivery",        "My order was supposed to arrive Monday and it's still not here."),
    ("other",           "Just wanted to say your support team last week was really helpful, thanks."),
    ("refund",          "You took money from me that you shouldn't have. I want it back."),
    ("account_access",  "I set up two-factor auth and now I can never get the code in time."),
]

CONFIDENCE_THRESHOLD = 0.8


def load_val_set(path="data/val.jsonl"):
    examples = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            user_content = record["messages"][0]["content"]
            text = user_content.split("Message: ", 1)[1].strip()
            label = record["messages"][1]["content"].strip()
            examples.append((label, text))
    return examples


# ── ModernBERT ────────────────────────────────────────────────────────────────

class ModernBERTPredictor:
    def __init__(self, model_dir):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def predict_batch(self, texts):
        enc = self.tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        confidences = probs.max(axis=-1)
        return [ID2LABEL[p] for p in preds], confidences.tolist()


# ── SetFit ────────────────────────────────────────────────────────────────────

class SetFitPredictor:
    def __init__(self, model_dir):
        self.model = SetFitModel.from_pretrained(model_dir)

    def predict_batch(self, texts):
        preds = self.model.predict(texts)
        probs = self.model.predict_proba(texts)
        if hasattr(probs, "numpy"):
            probs = probs.numpy()
        else:
            probs = np.array(probs)
        confidences = probs.max(axis=-1).tolist()
        pred_labels = [LABELS[p] if isinstance(p, (int, np.integer)) else p for p in preds]
        return pred_labels, confidences


# ── Eval core ─────────────────────────────────────────────────────────────────

def run_eval(predictor, examples):
    texts = [t for _, t in examples]
    true_labels = [l for l, _ in examples]
    pred_labels, confidences = predictor.predict_batch(texts)

    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", labels=LABELS, zero_division=0)

    below_threshold = [c for c in confidences if c < CONFIDENCE_THRESHOLD]
    pct_below = len(below_threshold) / len(confidences) * 100

    per_class_f1 = f1_score(true_labels, pred_labels, average=None, labels=LABELS, zero_division=0)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "pct_below_conf": pct_below,
        "per_class_f1": dict(zip(LABELS, per_class_f1)),
        "true_labels": true_labels,
        "pred_labels": pred_labels,
        "confidences": confidences,
    }


def print_confusion_matrix(results, title):
    true = results["true_labels"]
    pred = results["pred_labels"]
    cm = confusion_matrix(true, pred, labels=LABELS)
    print(f"\n{title}")
    header = f"{'':20}" + "".join(f"{l[:8]:>10}" for l in LABELS)
    print(header)
    for i, row_label in enumerate(LABELS):
        row = f"{row_label:<20}" + "".join(f"{cm[i][j]:>10}" for j in range(len(LABELS)))
        print(row)


def print_per_class(results):
    print(f"\n  {'Label':<20} {'F1':>6}")
    for label, f1 in results["per_class_f1"].items():
        print(f"  {label:<20} {f1:>6.3f}")


def evaluate_model(name, predictor, val_examples):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    print("\n[Synthetic val set]")
    syn = run_eval(predictor, val_examples)
    print(f"  Accuracy : {syn['accuracy']:.1%}")
    print(f"  Macro-F1 : {syn['macro_f1']:.3f}")
    print(f"  < {CONFIDENCE_THRESHOLD:.0%} conf : {syn['pct_below_conf']:.1f}% of examples")
    print_per_class(syn)
    print_confusion_matrix(syn, "  Confusion matrix (synthetic):")

    print("\n[Natural test set — n=10, ±30pt 95% CI]")
    nat = run_eval(predictor, NATURAL_TESTS)
    print(f"  Accuracy : {nat['accuracy']:.1%}")
    print(f"  Macro-F1 : {nat['macro_f1']:.3f}")
    print(f"  < {CONFIDENCE_THRESHOLD:.0%} conf : {nat['pct_below_conf']:.1f}% of examples")

    print("\n  Per-example:")
    for (expected, message), pred, conf in zip(NATURAL_TESTS, nat["pred_labels"], nat["confidences"]):
        mark = "✓" if pred == expected else "✗"
        print(f"  {mark} [{conf:.2f}] expected={expected:<16} got={pred:<16} | {message[:55]}")

    gap = syn["accuracy"] - nat["accuracy"]
    return {"name": name, "syn_acc": syn["accuracy"], "nat_acc": nat["accuracy"],
            "nat_f1": nat["macro_f1"], "gap": gap, "pct_below": nat["pct_below_conf"]}


def print_comparison_table(rows):
    print("\n" + "="*100)
    print("COMPARISON TABLE")
    print("="*100)
    header = f"{'Model':<35} {'Synthetic acc':>14} {'Natural acc':>12} {'Macro-F1 (nat)':>15} {'Gap':>6} {'% <0.8 conf':>12}"
    print(header)
    print("-"*100)

    # Pre-filled Qwen rows from README
    qwen_rows = [
        ("Qwen LoRA v1 (existing, w/ label hint)", 0.993, 0.60, "—", 0.393, "—"),
        ("Qwen LoRA v2 (existing, w/ label hint)", 0.972, 0.60, "—", 0.372, "—"),
    ]
    for name, syn, nat, f1, gap, below in qwen_rows:
        f1_str = f1 if isinstance(f1, str) else f"{f1:.3f}"
        below_str = below if isinstance(below, str) else f"{below:.1f}%"
        print(f"{name:<35} {syn:>13.1%} {nat:>11.1%} {f1_str:>15} {gap:>5.1%} {below_str:>12}")

    print("-"*100)
    for r in rows:
        f1_str = f"{r['nat_f1']:.3f}"
        below_str = f"{r['pct_below']:.1f}%"
        print(f"{r['name']:<35} {r['syn_acc']:>13.1%} {r['nat_acc']:>11.1%} {f1_str:>15} {r['gap']:>5.1%} {below_str:>12}")
    print("="*100)
    print("\nNote: Qwen uses an explicit label list in its inference prompt; ModernBERT/SetFit do not (correct for their architecture).")
    print("Note: Natural test n=10. Binomial 95% CI ≈ ±30 points at 60%. Treat natural results as directional, not conclusive.")


def main():
    val_examples = load_val_set()
    results = []

    models_to_eval = [
        ("ModernBERT v1", "modernbert", "models/modernbert/v1", ModernBERTPredictor),
        ("ModernBERT v2", "modernbert", "models/modernbert/v2", ModernBERTPredictor),
        ("SetFit v1 (full)", "setfit", "models/setfit/v1_full", SetFitPredictor),
        ("SetFit v1 (16-shot)", "setfit", "models/setfit/v1_fewshot", SetFitPredictor),
        ("SetFit v2 (full)", "setfit", "models/setfit/v2_full", SetFitPredictor),
    ]

    for name, _, model_dir, PredClass in models_to_eval:
        if not Path(model_dir).exists():
            print(f"\nSkipping {name} — {model_dir} not found (train it first)")
            continue
        predictor = PredClass(model_dir)
        row = evaluate_model(name, predictor, val_examples)
        results.append(row)

    if results:
        print_comparison_table(results)
    else:
        print("\nNo trained models found. Run modernbert_train.py and setfit_train.py first.")


if __name__ == "__main__":
    main()
