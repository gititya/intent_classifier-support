"""
Quick sanity check: run the fine-tuned model on hand-written, natural language
messages that were never in Bitext. One per class.
"""

from mlx_lm import load, generate

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
ADAPTER = "adapters/"

PROMPT_TEMPLATE = (
    "Classify this support message into one of: "
    "[billing, account_access, refund, product_how_to, bug_report, cancellation, delivery, other]."
    "\n\nMessage: {message}"
)

CHAT_TEMPLATE = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

# Hand-written natural messages — no Bitext templates, no placeholders
# Expected label listed alongside so we can check
TESTS = [
    ("billing",         "I just got my monthly statement and the numbers don't add up."),
    ("account_access",  "Locked out again — reset my password twice and it still won't let me in."),
    ("refund",          "I sent the item back two weeks ago and haven't heard anything about my money."),
    ("product_how_to",  "How do I actually use the bulk export feature? I can't figure it out."),
    ("bug_report",      "Every time I hit submit the whole page just goes white and nothing happens."),
    ("cancellation",    "I don't want to keep paying for this. How do I stop my subscription?"),
    ("delivery",        "My order was supposed to arrive Monday and it's still not here."),
    ("other",           "Just wanted to say your support team last week was really helpful, thanks."),
    # Two edge cases — slightly ambiguous
    ("refund",          "You took money from me that you shouldn't have. I want it back."),
    ("account_access",  "I set up two-factor auth and now I can never get the code in time."),
]

def predict(model, tokenizer, message):
    content = PROMPT_TEMPLATE.format(message=message)
    prompt = CHAT_TEMPLATE.format(content=content)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
    return response.strip().lower().split()[0] if response.strip() else "other"

def main():
    print(f"Loading fine-tuned model...")
    model, tokenizer = load(MODEL, adapter_path=ADAPTER)

    print(f"\n{'Expected':<20} {'Predicted':<20} {'':6} Message")
    print("-" * 90)

    correct = 0
    for expected, message in TESTS:
        predicted = predict(model, tokenizer, message)
        match = "✓" if predicted == expected else "✗"
        print(f"{expected:<20} {predicted:<20} {match}    {message}")
        if predicted == expected:
            correct += 1

    print(f"\nNatural language accuracy: {correct}/{len(TESTS)}")

if __name__ == "__main__":
    main()
