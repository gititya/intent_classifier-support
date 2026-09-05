# [EXPERIMENT] Intent Classifier

**Three model families, same 40–60% on natural language (n=10). The ceiling was the synthetic training data, not the model.**

Fine-tuned `Qwen2.5-1.5B-Instruct` to classify customer support messages into 8 labels: `billing`, `account_access`, `refund`, `product_how_to`, `bug_report`, `cancellation`, `delivery`, `other`.

## The one finding

Synthetic training data made the models overconfident.

The fine-tuned model hit 99.3% accuracy on 144 synthetic validation examples. Then I wrote ten messages myself—natural language, with no Bitext template patterns—and it dropped to 60% (n=10). The model learned surface keywords, not intent. "Cancel my subscription" → `cancellation`. "I don't want to keep paying for this" → `other`.

That gap is what the experiment was actually for.

## What I built and ran

**Five stages:**

1. **Data prep** — Downloaded 26K Bitext rows (Hugging Face), mapped 27 intents to 8 labels, capped at 100 per class to balance, then produced 576 training and 144 validation examples in chat JSONL format.
2. **LoRA config** — Rank 8, alpha 16, lr 1e-4, batch 4, 720 iters (~5 passes through the data).
3. **Training** — `mlx-lm` on M3 MacBook Air 16GB. 20 minutes. Peak memory: ~2GB. Loss: 1.28 → 0.15.
4. **Eval** — 99.3% on 144 synthetic validation examples. The one miss was an `other` message predicted as `account_access`.
5. **Baseline** — The same 144-example evaluation on the untouched base model scored 50.7%. The 48.6-point gain came from consistency, not new knowledge. The base model knew what "cancellation" meant; it just was not reliable about outputting exactly one label word.

**Then the ten-message natural-language test revealed the real number: 60%.**

## The augmentation round

Used Claude Haiku to paraphrase every training example without the obvious keyword. For `other`, generated fresh diverse examples instead — compliments, rants, off-topic questions — since paraphrasing Bitext `other` rows just produces more newsletter-unsubscribe variations. Dataset grew from 576 → 2,104 examples. Retrained (~75 minutes).

Result: still 60% on the same ten natural messages, but with different failures.

**Fixed:** `cancellation` ✓ — paraphrases without "cancel" worked. `other` ✓ — diverse generation worked.

**New failures:** `billing` and two `account_access` messages all predicted as `bug_report`. The augmented bug_report examples introduced "won't let me / doesn't work / can't" phrasing that overlaps with billing and account complaints. With only 14 original bug_report examples, the new data shifted that class boundary more than intended.

**The real fix:** Add bug_report examples that are clearly technical — error codes, stack traces, broken UI, crashes. That's a label definition problem, not a volume problem. More examples of the wrong kind made it worse, not better.

## The encoder round — was it the model?

The obvious next question: maybe a 1.5B decoder LLM is the wrong tool. So I built two more classifiers on the *same* data and tested them on the *same* 10 natural messages.

- **ModernBERT** (`answerdotai/ModernBERT-base`, 149M encoder) — the honest 2026 replacement for the old "small BERT baseline." Full fine-tune, v1 (576) and v2 (2,104).
- **SetFit** (`all-MiniLM-L6-v2` + logistic head) — contrastive few-shot, chosen because the bottleneck looked like small/overlapping-label data, not model size. Full, 16-shot, and v2.

The hypothesis was that SetFit's contrastive objective would resist keyword-memorization and close the gap.

**It didn't.** Every architecture landed in the same 40–60% natural-language band on ten messages. One different answer changes a score by ten points, so differences between adjacent results are noise.

| Model | Synthetic val (n=144) | Natural (n=10) | Macro-F1 (n=10) | Below 0.8 confidence (n=10) |
|---|---|---|---|---|
| Qwen LoRA v1 (w/ label hint) | 99.3% | 60% | — | — |
| Qwen LoRA v2 (w/ label hint) | 97.2% | 60% | — | — |
| ModernBERT v1 | 90.3% | 50% | 0.350 | 60% |
| ModernBERT v2 | 95.1% | 40% | 0.208 | 70% |
| SetFit v1 (full) | 96.5% | 50% | 0.362 | 60% |
| SetFit v1 (16-shot) | 89.6% | 20% | 0.125 | 100% |
| SetFit v2 (full) | 95.8% | 50% | 0.333 | 30% |

Two things fell out of this:

1. **More data made it worse.** ModernBERT went from 50% to 40% on the same ten natural messages when trained on about four times the augmented data. The Haiku paraphrases added more template-shaped variation, not natural diversity, so the models became more confident about the wrong keyword pattern.
2. **The confidence column is the useful output.** SetFit's 16-shot model put 100% of ten hard-case predictions below the 0.8 threshold—it knew it did not know. SetFit v2 put 30% of the same ten predictions below the threshold. This small test points toward routing high-confidence messages and sending the rest to a person; it does not validate that product decision.

**The conclusion: the ceiling is the data, not the model.** A 1.5B decoder, a 149M encoder, and a MiniLM plus logistic head all landed between 40% and 60% on the same ten natural messages. The eight-class space has genuine ambiguity (`billing` vs `bug_report`, `cancellation` vs `account_access`) that synthetic Bitext could not teach because it lacks keyword-free, naturally ambiguous messages. Full table and per-model failure patterns are in `EXPERIMENTS.md`.

(I also checked CFPB's 49K real consumer complaints as a real-language source — rejected: 77% credit-reporting, a taxonomy with no delivery/how-to/bug/cancellation, and `XXXX` redaction tokens that would just swap one template artifact for another.)

## What this is not

1. NOT a production classifier. Bitext is synthetic. The labels are generic e-commerce. The honest result is 40–60% on ten natural messages.
2. NOT a benchmark of Qwen2.5-1.5B, ModernBERT, or SetFit. A different dataset with tighter label definitions would produce a different result for all three.

## Stack

- Model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- Training: `mlx-lm` (LoRA, rank 8)
- Data: Bitext customer support dataset + Claude Haiku augmentation
- Hardware: Apple M3, 16GB unified memory
- No TypeScript, no server, no web UI

## Files

```
scripts/prepare_data.py     — download, map, balance, split
scripts/augment_data.py     — Haiku paraphrase augmentation
scripts/eval.py             — accuracy + confusion matrix (Qwen)
scripts/natural_test.py     — 10 hand-written messages, honest test
scripts/modernbert_train.py — ModernBERT full fine-tune (v1 + v2)
scripts/setfit_train.py     — SetFit contrastive (full, 16-shot, v2)
scripts/eval_compare.py     — unified harness: synthetic + natural, all models
config/lora_config.yaml     — training hyperparameters
EXPERIMENTS.md              — full comparison table + per-model failure patterns
```

## Results

Qwen LoRA round (the original experiment):

| Model | Synthetic val (n=144) | Natural language (n=10) |
|---|---|---|
| Baseline | 50.7% | — |
| v1 fine-tune (576 examples) | 99.3% | 60% |
| v2 fine-tune (2,104 examples, augmented) | 97.2% | 60% |

Full three-way comparison (Qwen vs ModernBERT vs SetFit) is in the table above and in `EXPERIMENTS.md`. Headline: everything landed at 40–60% on the same ten natural messages—**the ceiling is the data, not the model.**
