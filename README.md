# [EXPERIMENT] Intent Classifier

Fine-tuned `Qwen2.5-1.5B-Instruct` to classify customer support messages into 8 labels: `billing`, `account_access`, `refund`, `product_how_to`, `bug_report`, `cancellation`, `delivery`, `other`.

## The one finding

Synthetic training data makes you overconfident.

The fine-tuned model hit 99.3% accuracy on the validation set. Then I wrote 10 messages myself — natural language, no Bitext template patterns — and it dropped to 60%. The model learned surface keywords, not intent. "Cancel my subscription" → `cancellation`. "I don't want to keep paying for this" → `other`.

That gap is what the experiment was actually for.

## What I built and ran

**Five stages:**

1. **Data prep** — Downloaded 26K Bitext rows (Huggingface), mapped 27 intents to 8 labels, capped at 100 per class to balance, split 80/20. Output: 576 train, 144 val examples in chat JSONL format.
2. **LoRA config** — Rank 8, alpha 16, lr 1e-4, batch 4, 720 iters (~5 passes through the data).
3. **Training** — `mlx-lm` on M3 MacBook Air 16GB. 20 minutes. Peak memory: ~2GB. Loss: 1.28 → 0.15.
4. **Eval** — 99.3% on the synthetic val set. Confusion matrix clean — one `other` message predicted as `account_access`.
5. **Baseline** — Same eval on the untouched base model: 50.7%. The +48.6pp came from consistency, not new knowledge. The base model knew what "cancellation" meant; it just wasn't reliable about outputting exactly one label word.

**Then the natural language test revealed the real number: 60%.**

## The augmentation round

Used Claude Haiku to paraphrase every training example without the obvious keyword. For `other`, generated fresh diverse examples instead — compliments, rants, off-topic questions — since paraphrasing Bitext `other` rows just produces more newsletter-unsubscribe variations. Dataset grew from 576 → 2,104 examples. Retrained (~75 minutes).

Result: still 60% on natural language -different failures though.

**Fixed:** `cancellation` ✓ — paraphrases without "cancel" worked. `other` ✓ — diverse generation worked.

**New failures:** `billing` and two `account_access` messages all predicted as `bug_report`. The augmented bug_report examples introduced "won't let me / doesn't work / can't" phrasing that overlaps with billing and account complaints. With only 14 original bug_report examples, the new data shifted that class boundary more than intended.

**The real fix:** Add bug_report examples that are clearly technical — error codes, stack traces, broken UI, crashes. That's a label definition problem, not a volume problem. More examples of the wrong kind made it worse, not better.

## What this is not

1. NOT a production classifier. Bitext is synthetic. The labels are generic e-commerce. The 60% on natural language is the honest number.
2. NOT a benchmark of Qwen2.5-1.5B. A different dataset with tighter label definitions would produce a different result.

## Stack

- Model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- Training: `mlx-lm` (LoRA, rank 8)
- Data: Bitext customer support dataset + Claude Haiku augmentation
- Hardware: Apple M3, 16GB unified memory
- No TypeScript, no server, no web UI

## Files

```
scripts/prepare_data.py   — download, map, balance, split
scripts/augment_data.py   — Haiku paraphrase augmentation
scripts/eval.py           — accuracy + confusion matrix
scripts/natural_test.py   — 10 hand-written messages, honest test
config/lora_config.yaml   — training hyperparameters
```

## Results

| Model | Synthetic val | Natural language |
|---|---|---|
| Baseline | 50.7% | — |
| v1 fine-tune (576 examples) | 99.3% | 60% |
| v2 fine-tune (2,104 examples, augmented) | 97.2% | 60% |
