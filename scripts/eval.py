"""
Stage 4: Evaluate the fine-tuned model on the validation set.

Run after training:
  python3 scripts/eval.py --adapter adapters/
  python3 scripts/eval.py            (baseline: no adapter, base model only)
"""

import argparse
import json
from collections import defaultdict

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

LABEL_SET = set(LABELS)

VAL_PATH = "data/val.jsonl"
MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def load_val():
    examples = []
    with open(VAL_PATH) as f:
        for line in f:
            obj = json.loads(line)
            user_msg = obj["messages"][0]["content"]
            true_label = obj["messages"][1]["content"].strip()
            examples.append((user_msg, true_label))
    return examples


def predict(model, tokenizer, prompt, adapter_path=None):
    from mlx_lm import generate
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=10,  # label is at most 2 words — no need for more
        verbose=False,
    )
    # Strip and take only the first token/word — we want exactly the label
    return response.strip().lower().split()[0] if response.strip() else "other"


def build_prompt(user_content):
    # Use the same chat template the model was trained with
    return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="Path to adapter folder (omit for baseline)")
    args = parser.parse_args()

    from mlx_lm import load
    print(f"Loading model: {MODEL}")
    if args.adapter:
        print(f"  with adapter: {args.adapter}")
        model, tokenizer = load(MODEL, adapter_path=args.adapter)
    else:
        print("  NO adapter — baseline run")
        model, tokenizer = load(MODEL)

    examples = load_val()
    print(f"\nRunning inference on {len(examples)} validation examples...")

    correct = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))  # confusion[true][predicted]

    for i, (prompt_text, true_label) in enumerate(examples):
        prompt = build_prompt(prompt_text)
        predicted = predict(model, tokenizer, prompt, args.adapter)

        # If model outputs something outside our label set, count as wrong
        if predicted not in LABEL_SET:
            predicted = "other"

        per_class_total[true_label] += 1
        confusion[true_label][predicted] += 1

        if predicted == true_label:
            correct += 1
            per_class_correct[true_label] += 1

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(examples)} done...")

    total = len(examples)
    accuracy = correct / total

    print(f"\n{'='*50}")
    label = "FINE-TUNED" if args.adapter else "BASELINE (no adapter)"
    print(f"Results — {label}")
    print(f"{'='*50}")
    print(f"Overall accuracy: {accuracy:.1%}  ({correct}/{total})\n")

    print("Per-class accuracy:")
    for lbl in LABELS:
        t = per_class_total[lbl]
        c = per_class_correct[lbl]
        bar = "#" * c + "." * (t - c)
        pct = f"{c/t:.1%}" if t > 0 else "N/A"
        print(f"  {lbl:<20} {pct:>6}  [{bar}] {c}/{t}")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = f"{'':20}" + "".join(f"{l[:6]:>8}" for l in LABELS)
    print(header)
    for true_lbl in LABELS:
        row = f"{true_lbl:<20}"
        for pred_lbl in LABELS:
            count = confusion[true_lbl][pred_lbl]
            cell = str(count) if count > 0 else "."
            row += f"{cell:>8}"
        print(row)

    print("\nRead the confusion matrix: each row is the true label.")
    print("Numbers off the diagonal are misclassifications.")
    print("Big off-diagonal numbers = label boundary confusion.\n")


if __name__ == "__main__":
    main()
