"""
Stage 1: Download Bitext, map to our 8 labels, balance, split, write JSONL.

Run: python3 scripts/prepare_data.py
Outputs: data/train.jsonl, data/val.jsonl
"""

import json
import random
from collections import defaultdict
from datasets import load_dataset

# The 8 labels we want the classifier to predict.
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

# Bitext has 27 intents. We map them to our 8 labels.
# Any intent not listed here gets mapped to "other".
# Source: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
BITEXT_INTENT_MAP = {
    # billing
    "check_invoice": "billing",
    "get_invoice": "billing",
    "check_payment_methods": "billing",
    "payment_issue": "billing",

    # account_access
    "recover_password": "account_access",
    "registration_problems": "account_access",
    "edit_account": "account_access",
    "delete_account": "account_access",
    "switch_account": "account_access",

    # refund
    "get_refund": "refund",
    "track_refund": "refund",

    # product_how_to
    "contact_customer_service": "product_how_to",
    "review": "product_how_to",
    "set_up_shipping_address": "product_how_to",

    # cancellation
    "cancel_order": "cancellation",
    "cancel_service": "cancellation",

    # delivery
    "delivery_options": "delivery",
    "delivery_period": "delivery",
    "track_order": "delivery",
    "place_order": "delivery",
    "change_order": "delivery",
    "complaint": "delivery",

    # bug_report — Bitext has no direct bug reports; intentionally sparse
    # We'll handle this below with a small manual set

    # other (explicit catch)
    "newsletter_subscription": "other",
    "check_cancellation_fee": "other",
    "contact_human_agent": "other",
}

# Small manual examples for bug_report since Bitext doesn't cover it.
# Real-world classifiers always have coverage gaps — patching with manual data is normal.
MANUAL_BUG_REPORTS = [
    "The app crashes every time I open it on my phone.",
    "I keep getting a 500 error when I try to submit the form.",
    "The page won't load — it just spins forever.",
    "Your checkout button doesn't work on Safari.",
    "I get an error message that says 'undefined' when I log in.",
    "The search bar returns no results even for things I know exist.",
    "Dark mode is broken — the text is invisible.",
    "I uploaded my photo but it never appears on my profile.",
    "The export button downloads a blank file every time.",
    "Notifications stopped working after your last update.",
    "The mobile app logged me out and won't let me back in.",
    "I can't complete checkout — it loops back to the cart.",
    "The video player freezes at exactly 2 minutes every time.",
    "My order history shows duplicates of every item.",
    "The map on your store locator doesn't zoom.",
    "Filter options disappear after I apply them.",
    "Two-factor auth code never arrives via SMS.",
    "The date picker accepts invalid dates like Feb 30th.",
    "My saved address was deleted after your maintenance window.",
    "The app shows my balance as $0 even though I have funds.",
]

PROMPT_TEMPLATE = (
    "Classify this support message into one of: "
    "[billing, account_access, refund, product_how_to, bug_report, cancellation, delivery, other]."
    "\n\nMessage: {message}"
)

MAX_PER_CLASS = 100
VAL_RATIO = 0.2
SEED = 42


def format_example(message, label):
    return {
        "messages": [
            {"role": "user", "content": PROMPT_TEMPLATE.format(message=message.strip())},
            {"role": "assistant", "content": label},
        ]
    }


def main():
    random.seed(SEED)

    print("Downloading Bitext dataset from HuggingFace...")
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    print(f"  Total rows: {len(ds)}")

    # Bucket by our label
    buckets = defaultdict(list)
    skipped = 0
    for row in ds:
        intent = row["intent"].lower().strip()
        label = BITEXT_INTENT_MAP.get(intent, "other")
        buckets[label].append(row["instruction"])

    # Add manual bug_report examples
    for msg in MANUAL_BUG_REPORTS:
        buckets["bug_report"].append(msg)

    # Show distribution before balancing
    print("\nRaw counts per label (before balancing):")
    for label in LABELS:
        print(f"  {label:<20} {len(buckets[label])}")

    # Balance: cap each class at MAX_PER_CLASS, shuffle first
    examples = []
    for label in LABELS:
        items = buckets[label]
        random.shuffle(items)
        items = items[:MAX_PER_CLASS]
        for msg in items:
            examples.append(format_example(msg, label))

    random.shuffle(examples)

    # Train / val split
    split_idx = int(len(examples) * (1 - VAL_RATIO))
    train_set = examples[:split_idx]
    val_set = examples[split_idx:]

    print(f"\nAfter balancing:")
    print(f"  Total examples : {len(examples)}")
    print(f"  Train          : {len(train_set)}")
    print(f"  Val            : {len(val_set)}")

    # Write JSONL — mlx-lm expects one JSON object per line
    for path, split in [("data/train.jsonl", train_set), ("data/val.jsonl", val_set)]:
        with open(path, "w") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")
        print(f"  Wrote {path}")

    # Sanity check: print one example from each split
    print("\n--- Sample train example ---")
    print(json.dumps(train_set[0], indent=2))
    print("\n--- Sample val example ---")
    print(json.dumps(val_set[0], indent=2))
    print("\nDone. Next: python3 scripts/check_data.py to verify label balance.")


if __name__ == "__main__":
    main()
