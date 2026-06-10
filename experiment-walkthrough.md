# Fine-Tuning Experiment Walkthrough
### Intent Classifier — Qwen2.5-1.5B on Customer Support Messages

> **How to use this as a template:** The structure (5 stages, each with Why / What / What to observe) applies to any supervised fine-tuning experiment. Swap the model, task, and dataset. The mechanics are the same.

---

## Before you start — three concepts to get straight

Fine-tuning sits between two other things people mean when they say "training a model."

**Pre-training** is where a model learns language itself — grammar, facts, reasoning — from billions of tokens. Done by companies like Alibaba or Meta over months on thousands of GPUs. You never do this.

**Fine-tuning** is where you take a pre-trained model and show it examples of a specific task. The model already understands language. You're teaching it to do *one job consistently*. This takes minutes to hours on a laptop. This is what this experiment is.

**Inference** is just running the model to get an output. After fine-tuning, you run inference on examples the model never saw during training to measure whether it actually learned.

---

## The five stages

---

### Stage 1 — Data preparation

**Why this stage exists:**
The model learns from examples, not from instructions. Every example must be in a specific format — a chat template with a user message and the expected assistant response. If the format is wrong, training either fails or produces garbage. Getting data right before training starts is non-negotiable.

There are two sub-problems inside data prep:

- **Label mapping.** External datasets (like Bitext) have their own category names. You need to map them to your labels. Some of your labels may be sparse in the external data — you'll need to patch manually.
- **Label balance.** If one class has 5,000 examples and another has 20, the model learns to guess the big class. Cap every class at the same number before training.

**What we did:**
Downloaded the Bitext customer support dataset (26K rows, 27 intents). Mapped those 27 intents down to 8 labels: `billing`, `account_access`, `refund`, `product_how_to`, `bug_report`, `cancellation`, `delivery`, `other`. The `bug_report` class barely existed in Bitext — patched with 20 hand-written examples. Capped every class at 100 examples. Output format:

```jsonl
{"messages": [
  {"role": "user", "content": "Classify this support message into one of: [billing, account_access, ...]\n\nMessage: I can't log in after resetting my password."},
  {"role": "assistant", "content": "account_access"}
]}
```

**Train/val split (80/20):**
576 training examples, 144 validation examples. The model only learns from training. Validation is held back entirely — used only to measure results after training. Think of it as practice questions vs. the real exam.

**What to observe:**
Run the script and look at the raw counts before balancing. If one class dominates (e.g. 5,000 vs 20), that's your signal to balance aggressively. After balancing, check that all 8 classes are represented in both train and val.

---

### Stage 2 — LoRA config

**Why LoRA instead of full fine-tuning:**
Full fine-tuning updates every weight in the model — forward pass, backward pass, gradients for 1.5 billion parameters. That requires ~12–16GB of memory just for training state, on top of the model itself.

LoRA (Low-Rank Adaptation) freezes all original weights and trains only a small side-file called an adapter. The base model is untouched. Only 0.171% of parameters are trained — 2.6M out of 1.5B. Peak memory during training stays under 2GB.

**The key knobs and what they mean:**

| Parameter | Value used | What it does |
|---|---|---|
| `lora_rank` | 8 | How many extra parameters per layer. Rank 8 is the standard starting point for classification. Higher = more capacity, more overfitting risk. |
| `lora_alpha` | 16 | Scales the adapter's influence. Rule of thumb: alpha = 2× rank. |
| `lora_dropout` | 0.05 | Light regularisation. Reduces overfitting on small datasets. |
| `learning_rate` | 1e-4 | How large each weight update step is. Standard LoRA starting point. Too high = unstable loss. |
| `batch_size` | 4 | Examples per gradient update. Safe for 16GB. |
| `iters` | 720 | Total training steps. At batch size 4 and 576 examples, this is ~5 passes through the data. |

**What to observe:**
Nothing visible yet — this stage is just configuration. The knobs matter during training when you watch the loss curve.

---

### Stage 3 — Training

**Why the loss curve is the thing to watch:**
Loss measures how wrong the model is. Lower = the model is more confident it's predicting the right label. You watch two numbers: train loss and val loss (if validation is wired up). They tell different stories.

- **Train loss dropping steadily** = model is learning the task. Expected.
- **Val loss dropping alongside train loss** = the learning is generalising to unseen examples. Good.
- **Val loss rising while train loss keeps dropping** = overfitting. The model is memorising training examples rather than learning the underlying pattern. Fix: fewer epochs, more data, higher dropout.

**What happened in this experiment:**

| Iter | Train loss | What it means |
|---|---|---|
| 10 | 1.282 | Model is essentially guessing randomly. High loss is expected at the start. |
| 20 | 0.487 | Big drop. Model learned the basic structure fast — output one label word, nothing else. |
| 300 | 0.171 | Continued refinement. Model is getting confident. |
| 500–720 | ~0.14–0.15 | Loss flattened. Not going to zero — that would mean memorisation. Healthy plateau. |

Peak memory: **1.993 GB** — the LoRA advantage, demonstrated.

**What to observe:**
If loss drops fast then flattens, that's healthy for a small classification dataset. If loss keeps dropping all the way to near zero, you're memorising. If loss bounces around without a trend, your learning rate may be too high.

---

### Stage 4 — Eval on the fine-tuned model

**Why you need a separate eval script, not just training loss:**
Training loss tells you the model got better at predicting training examples. It doesn't tell you it generalised. The eval script runs inference on 144 examples the model never saw, compares predicted label vs. correct label, and prints accuracy and a confusion matrix.

**Accuracy alone isn't enough.** A model that predicts `delivery` for every message would have 10.4% accuracy on a balanced dataset — looks like it's doing something. The confusion matrix shows *where* it's going wrong. That's the real diagnostic.

**How to read the confusion matrix:**
Rows are the true label. Columns are what the model predicted. Numbers on the diagonal = correct. Numbers off the diagonal = misclassifications. A cluster of off-diagonal numbers between two labels means the model is confused about the boundary between them.

**Results — fine-tuned model:**
```
Overall accuracy: 99.3%  (143/144)

billing              100%    account_access  100%
refund               100%    product_how_to  100%
bug_report           100%    cancellation    100%
delivery             100%    other            95%
```
One `other` message was predicted as `account_access`. Everything else: perfect.

**What to observe:**
Any class with low per-class accuracy deserves attention. That's where your label boundary is fuzzy, your examples are too similar to another class, or your manual data was too homogeneous.

---

### Stage 5 — Baseline comparison

**Why this stage is mandatory:**
This is the step most people skip, and it's the most important one. Before calling the fine-tune a success, you run the exact same eval on the base model with no adapter. If the base model was already 95% accurate with a good prompt, you didn't need to fine-tune — you needed a better prompt.

Fine-tuning has a cost: it takes time to run, requires a dataset, and the adapter is another artefact to maintain. That cost is only worth it if the fine-tuned model does something the base model genuinely can't.

**Results — baseline (no adapter):**
```
Overall accuracy: 50.7%  (73/144)

billing       45.8%    account_access  38.1%
refund        93.8%    product_how_to  14.3%
bug_report    66.7%    cancellation    95.2%
delivery      26.7%    other           40.0%
```

The base model knows what these labels mean — `cancellation` and `refund` are 93–95% because those words appear almost verbatim in the message. But `product_how_to` at 14.3% and `delivery` at 26.7% are near-random. The base model is also inconsistent: it sometimes outputs a full sentence instead of just the label word.

**The +48.6 percentage point gain** came entirely from consistency and precision, not new knowledge. Fine-tuning drilled in: output exactly one label word, nothing else, every time.

**What to observe:**
If your baseline is already 90%+, reconsider whether fine-tuning was necessary. If your baseline is 50% or below, fine-tuning is clearly buying you something real.

---

## Summary of results

| Model | Accuracy | Notes |
|---|---|---|
| Baseline (no adapter) | 50.7% | Inconsistent output format; weak on vague label boundaries |
| v1 fine-tune — synthetic val set | 99.3% | Precise, consistent, exactly one label per message |
| v1 fine-tune — natural language test | 60.0% | Honest generalisation score; exposed by hand-written messages |
| v2 fine-tune — synthetic val set | 97.2% | Slightly lower; augmented data introduced more label boundary overlap |
| v2 fine-tune — natural language test | 60.0% | Same headline number, different failures |

**Peak training memory:** ~2GB  
**Training time:** ~20 minutes on M3 MacBook Air 16GB  
**Trainable parameters:** 2.638M / 1,543.714M (0.171%)  
**Dataset size:** 720 examples (576 train / 144 val), 8 balanced classes

---

## Bonus stage — Natural language test

**Why this matters:**
After seeing 99.3% on the validation set, it would be easy to call the experiment a success and move on. But the validation set came from the same synthetic Bitext dataset as the training set — same phrasing patterns, same `{{Order Number}}` placeholders, same generation style. The model was tested on data that looked almost identical to what it trained on.

A more honest test is to write 10 messages yourself, in completely natural language, and run them through the model cold.

**Results — 10 hand-written natural language messages:**

| Expected | Predicted | Message |
|---|---|---|
| billing | billing ✓ | I just got my monthly statement and the numbers don't add up. |
| account_access | account_access ✓ | Locked out again — reset my password twice and it still won't let me in. |
| refund | refund ✓ | I sent the item back two weeks ago and haven't heard anything about my money. |
| product_how_to | **other ✗** | How do I actually use the bulk export feature? I can't figure it out. |
| bug_report | bug_report ✓ | Every time I hit submit the whole page just goes white and nothing happens. |
| cancellation | **other ✗** | I don't want to keep paying for this. How do I stop my subscription? |
| delivery | delivery ✓ | My order was supposed to arrive Monday and it's still not here. |
| other | **product_how_to ✗** | Just wanted to say your support team last week was really helpful, thanks. |
| refund | refund ✓ | You took money from me that you shouldn't have. I want it back. |
| account_access | **bug_report ✗** | I set up two-factor auth and now I can never get the code in time. |

**Natural language accuracy: 6/10 (60%)**

**What the misses reveal:**

- `cancellation` → `other`: "I don't want to keep paying" — the word "cancel" never appears. The model learned the keyword, not the intent behind it. Bitext examples nearly always contained "cancel" explicitly.
- `product_how_to` → `other`: "bulk export feature" — too specific and technical. Training examples were generic how-to questions. The model didn't recognise a feature name as a how-to trigger.
- `other` → `product_how_to`: A compliment ("your support team was really helpful"). Nothing in training data looked like praise. The model had no signal for what genuine `other` looks like, so it guessed.
- `account_access` → `bug_report`: "2FA code never arrives" — genuinely ambiguous. A human might argue either label. This is a real label boundary problem, not a model failure.

**The lesson:**
Synthetic training data makes you overconfident. 99.3% on a synthetic val set and 60% on natural language is the real story of this experiment. The gap isn't because fine-tuning failed — it's because the training data was too clean, too keyword-heavy, and too uniform. The model learned surface patterns, not intent.

This is one of the most important things to learn from a first fine-tuning experiment, and you only see it if you run the natural language test.

---

## Bonus stage — LLM augmentation (v2)

**What we did:**
Used Claude Haiku to generate 3 paraphrases of every training example, specifically instructed to avoid the obvious keyword for each class. For `other`, generated fresh diverse examples — compliments, rants, off-topic questions, ambiguous messages — since paraphrasing existing Bitext `other` examples just produces more of the same. Dataset grew from 576 → 2,104 examples. Retrained with the same config and ran both evals again.

**Results:**

| Test | v1 | v2 |
|---|---|---|
| Synthetic val accuracy | 99.3% | 97.2% |
| Natural language accuracy | 60% | 60% |

**What changed under the hood:**

v1 failures: `cancellation`, `product_how_to`, `other`, `account_access`

v2 failures: `billing`, `product_how_to`, `account_access` × 2

Two fixes: `cancellation` ✓ (paraphrases without "cancel" worked), `other` ✓ (fresh diverse generation worked).

Two new failures: `billing` and extra `account_access` both predicted as `bug_report`. The augmented `bug_report` examples introduced phrases like "won't let me", "doesn't work", "can't" — which overlap with billing and account complaints. With only 14 original bug_report examples, the augmented set shifted that class's boundary more than the others.

**The real fix from here:**
Add explicit bug_report examples that are clearly technical — stack traces, error codes, broken UI, crashes — so the model stops treating any "something isn't working" message as a bug report. This is a label definition problem, not a data volume problem. More examples of the wrong kind make it worse.

**What augmentation actually taught:**
Data quality determines generalisation more than training iterations. The 60% ceiling is a label boundary problem — some of these classes overlap in natural language in ways that Bitext's keyword-heavy generation obscures. Fixing it properly means either tightening the label definitions or accepting that a 1.5B model on 700–2,000 synthetic examples has a real ceiling on ambiguous boundary cases.

---

## What this experiment teaches

1. **Data formatting is where most time goes** — not training. Getting the JSONL right, balancing classes, handling sparse labels — that's the real work.
2. **LoRA is the right default for small classification tasks on consumer hardware** — 2GB peak memory vs. 12GB+ for full fine-tuning.
3. **Loss flattening is healthy** — a plateau means the model learned what it can from this data. Near-zero loss means memorisation.
4. **Always run the baseline** — the gap between 50.7% and 99.3% tells you exactly what fine-tuning bought. No gap = you didn't need to fine-tune.
5. **The confusion matrix is more useful than accuracy** — it shows you which label boundaries are fuzzy and where to add more training data next.
6. **Synthetic training data makes you overconfident** — always test on natural, hand-written examples after seeing a high val accuracy. 99.3% on synthetic data, 60% on real language, is the honest result of this experiment. The gap is a data quality problem, not a model problem.
7. **Augmentation fixes keyword dependence, not label boundary fuzziness** — paraphrasing without keywords fixed the `cancellation` miss. It didn't fix `account_access` vs `bug_report` because that's a genuine semantic overlap, not a surface-word problem. Know which problem you're solving before augmenting.
