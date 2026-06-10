# CLAUDE.md — intent_classifier

## What this project is

A fine-tuning learning exercise. The goal is to fine-tune `Qwen2.5-1.5B-Instruct` to classify customer support messages into one of 8 fixed labels. This is a mechanics kata — the point is to learn dataset formatting, train/eval split, LoRA config, inference, and eval, not to ship a product.

Reference: `mlx-fine-tuning-experiments.md` → "Optional Kata: Dead-Simple Intent Classifier"

## Labels

`billing`, `account_access`, `refund`, `product_how_to`, `bug_report`, `cancellation`, `delivery`, `other`

## Model

`Qwen2.5-1.5B-Instruct` (fits comfortably in 16GB unified memory)

## Method

SFT (supervised fine-tuning) via `mlx-lm`. LoRA optional — try SFT first at this size.

## Stack

- Python 3.11
- `mlx-lm` (already installed)
- `datasets` (HuggingFace) for Bitext seed data
- No TypeScript, no web UI, no server

## Project layout

```
intent_classifier/
  data/
    raw/           # Bitext download + any manual examples
    train.jsonl    # 80% split, chat-template format
    val.jsonl      # 20% split
  adapters/        # LoRA adapter weights output by mlx-lm
  scripts/
    prepare_data.py   # download Bitext, filter to our 8 labels, format as chat JSONL
    eval.py           # run inference on val set, print accuracy + confusion matrix
  config/
    lora_config.yaml  # mlx-lm LoRA hyperparameters
  CLAUDE.md
  SKILL.md
```

## Success state

Model outputs the exact label string with no explanation or JSON. Accuracy tracked per-class. Confusion matrix printed by eval script.

## What we are NOT doing

- No web app or API wrapper
- No deployment
- No TypeScript
- No fancy prompt chaining — one message in, one label out

## File integrity

CLAUDE.md and SKILL.md are append-only. No wholesale rewrites.
