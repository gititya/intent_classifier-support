"""
Stage 1b: Augment training data using Claude API.

For the 7 content classes: generate 3 paraphrases per example without the
obvious keyword, so the model learns intent not surface words.

For the `other` class: generate new diverse examples (compliments, rants,
off-topic questions, ambiguous messages) since paraphrasing existing Bitext
`other` examples just produces more of the same newsletter/catch-all content.

Model: claude-haiku-4-5-20251001 (fast, cheap — sufficient for paraphrasing)
Key: read from macOS keychain at runtime, never hardcoded.

Run: python3 scripts/augment_data.py
Outputs: data/train_augmented.jsonl (original train + augmented examples)
         data/val.jsonl is unchanged — we never augment the eval set
"""

import json
import random
import subprocess
import anthropic
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

# Keywords that tend to appear in Bitext examples per class.
# We tell the model to avoid these so paraphrases don't just
# swap synonyms for the same surface word.
KEYWORDS_TO_AVOID = {
    "billing":        ["invoice", "statement", "bill", "charge", "payment"],
    "account_access": ["password", "login", "log in", "locked", "access"],
    "refund":         ["refund", "money back", "return"],
    "product_how_to": ["how do I", "how to", "instructions", "guide"],
    "bug_report":     ["crash", "error", "broken", "not working", "bug"],
    "cancellation":   ["cancel", "cancellation", "unsubscribe"],
    "delivery":       ["delivery", "shipping", "arrive", "track", "order"],
}

# For `other`, we don't paraphrase — we generate fresh diverse examples.
OTHER_TYPES = [
    "a genuine compliment praising the company or support team",
    "a general rant expressing frustration with the company without a specific issue",
    "an off-topic question (referral program, opening hours, phone number, social media)",
    "a message that is ambiguous and doesn't clearly fit any support category",
    "a very short or vague message like 'hello?' or 'just checking in'",
]

PARAPHRASE_PROMPT = """\
You are helping build a training dataset for a customer support intent classifier.

Rewrite the following support message 3 times. Each rewrite must:
- Express the same intent as the original
- Avoid these words: {keywords}
- Sound like a real customer message — natural, varied tone
- Be a single sentence or two at most
- NOT explain or add context

Original message: {message}

Output exactly 3 rewrites, one per line, no numbering, no labels, no extra text."""

OTHER_GENERATION_PROMPT = """\
You are helping build a training dataset for a customer support intent classifier.
The `other` class covers messages that don't fit billing, account access, refunds,
how-to questions, bug reports, cancellations, or delivery.

Generate {n} different customer messages that are: {message_type}

Rules:
- Each message should sound like a real customer wrote it
- Vary the tone and phrasing across messages
- One message per line, no numbering, no labels, no extra text
- Short — one or two sentences each"""

PARAPHRASES_PER_EXAMPLE = 3
OTHER_EXAMPLES_PER_TYPE = 8
SEED = 42


def get_api_key():
    result = subprocess.run(
        ["security", "find-generic-password", "-s", "anthropic", "-w"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    result = subprocess.run(
        ["security", "find-generic-password", "-s", "ANTHROPIC_API_KEY", "-w"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    raise RuntimeError("Anthropic API key not found in keychain. Store it with: security add-generic-password -s anthropic -a anthropic -w YOUR_KEY")


def load_train():
    examples = []
    with open("data/train.jsonl") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def extract_message(example):
    content = example["messages"][0]["content"]
    # Message is after the last "\n\nMessage: "
    return content.split("\n\nMessage: ")[-1].strip()


def format_example(message, label):
    prompt = (
        "Classify this support message into one of: "
        "[billing, account_access, refund, product_how_to, bug_report, cancellation, delivery, other]."
        f"\n\nMessage: {message.strip()}"
    )
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": label},
        ]
    }


def paraphrase(client, message, label):
    keywords = KEYWORDS_TO_AVOID.get(label, [])
    keywords_str = ", ".join(keywords) if keywords else "none"
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": PARAPHRASE_PROMPT.format(
                keywords=keywords_str,
                message=message,
            )
        }]
    )
    lines = response.content[0].text.strip().split("\n")
    return [l.strip() for l in lines if l.strip()][:PARAPHRASES_PER_EXAMPLE]


def generate_other_examples(client, message_type, n):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": OTHER_GENERATION_PROMPT.format(
                n=n,
                message_type=message_type,
            )
        }]
    )
    lines = response.content[0].text.strip().split("\n")
    return [l.strip() for l in lines if l.strip()][:n]


def main():
    random.seed(SEED)

    print("Reading API key from keychain...")
    api_key = get_api_key()
    client = anthropic.Anthropic(api_key=api_key)

    print("Loading training data...")
    train_examples = load_train()
    print(f"  Original training examples: {len(train_examples)}")

    # Group by label so we can process class by class
    by_label = defaultdict(list)
    for ex in train_examples:
        label = ex["messages"][1]["content"]
        by_label[label].append(ex)

    augmented = []

    # --- Paraphrase the 7 content classes ---
    for label in LABELS:
        if label == "other":
            continue

        examples = by_label[label]
        print(f"\nAugmenting '{label}' ({len(examples)} examples × {PARAPHRASES_PER_EXAMPLE} paraphrases)...")

        for i, ex in enumerate(examples):
            message = extract_message(ex)
            try:
                paraphrases = paraphrase(client, message, label)
                for p in paraphrases:
                    augmented.append(format_example(p, label))
            except Exception as e:
                print(f"  Warning: failed on example {i}: {e}")
                continue

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(examples)} done, {len(augmented)} augmented so far")

    # --- Generate fresh `other` examples ---
    print(f"\nGenerating fresh 'other' examples ({len(OTHER_TYPES)} types × {OTHER_EXAMPLES_PER_TYPE})...")
    for msg_type in OTHER_TYPES:
        try:
            examples = generate_other_examples(client, msg_type, OTHER_EXAMPLES_PER_TYPE)
            for ex in examples:
                augmented.append(format_example(ex, "other"))
            print(f"  Generated {len(examples)} examples for: {msg_type[:60]}...")
        except Exception as e:
            print(f"  Warning: failed for type '{msg_type[:40]}': {e}")

    # Combine original + augmented, shuffle
    all_examples = train_examples + augmented
    random.shuffle(all_examples)

    output_path = "data/train_augmented.jsonl"
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n{'='*50}")
    print(f"Original training examples : {len(train_examples)}")
    print(f"Augmented examples added   : {len(augmented)}")
    print(f"Total                      : {len(all_examples)}")
    print(f"Written to                 : {output_path}")
    print(f"\nNext: retrain using data/train_augmented.jsonl")
    print(f"Rename it to data/train.jsonl before running mlx_lm.lora")


if __name__ == "__main__":
    main()
