# Teaching Style Guide
### For Technical Learning Experiments

> This is not about what to teach. It's about how to teach it — the principles behind explaining technical work to someone learning by doing. Apply this across any experiment: fine-tuning, evals, agents, infra, whatever.

---

## The core premise

The learner is not reading documentation. They are watching something happen on their machine, in real time, with real output. Every explanation should be anchored to something they can see, run, or observe. Abstract principles land when they're attached to a concrete moment.

If you explain a concept before they've seen the thing it describes, it won't stick. If you explain it while they're looking at the output, it will.

---

## Principle 1 — Why before what

Never explain what a thing is before explaining why it exists.

"LoRA is a method that adds low-rank matrices to frozen model layers" is a what. It means nothing to someone who doesn't know why freezing the base model matters.

"Full fine-tuning requires 12GB of memory just for training state, which your machine doesn't have. LoRA trains only a small side-file and uses 2GB. That's why we use it" — that's a why. The what follows naturally.

**The test:** Can the learner answer "why do we do it this way and not some other way?" If not, the why wasn't clear enough.

---

## Principle 2 — Stage the journey before you start

Before doing anything, give the learner a map of all the stages. Not every detail — just the names and the sequence. This does two things:

1. The learner knows where they are at all times. They're not just following instructions blindly.
2. When something fails or takes longer than expected, they can orient themselves. "We're still in Stage 1, Stage 3 is when the interesting things happen."

Format that works:
```
Stage 1 — Data prep
Stage 2 — Config
Stage 3 — Training
Stage 4 — Eval
Stage 5 — Baseline comparison
```

You don't need to explain all five at once. Just name them. Come back to each one when you arrive at it.

---

## Principle 3 — Name the failure before it happens

Don't only teach the happy path. The learner will hit failure modes. If they've never heard of a failure mode before they see it, it looks like a personal mistake. If you named it in advance, it's a known landmark.

Before training: "Watch for val loss rising while train loss keeps dropping — that's overfitting. It's expected at this dataset size. It's not a failure, it's the lesson."

Before the baseline eval: "If the baseline is already 90%, the fine-tune wasn't necessary. That outcome is as useful as a 99% fine-tune — it tells you prompting was sufficient."

**Why this matters:** Named failure modes feel like progress. Unnamed failure modes feel like you broke something.

---

## Principle 4 — Explain the output while it's on screen

Don't wait until the end to explain what happened. When training output appears, explain the numbers in real time.

"Train loss 1.282 → 0.487 in 10 steps — that big drop is the model learning the task structure fast. It figured out 'output one label word' before it figured out which label."

This is the most valuable teaching moment in any experiment: the learner is looking at real numbers from their real machine. That's when abstractions become concrete. Don't waste it by staying silent.

---

## Principle 5 — Use analogies grounded in things the learner already knows

The 80/20 train/val split is abstract. "Practice questions vs. the real exam" is not. The learner immediately knows: you study from practice questions, but the real exam has different ones. If you memorised answers without understanding, you fail the real exam.

The rule: find the thing the learner already knows that has the same structure as the thing you're explaining. You're not dumbing it down — you're building a bridge.

Good analogies to reach for in ML contexts:
- Train/val split → practice exam vs. real exam
- Overfitting → student who memorised answers instead of understanding
- Loss → how wrong the model is (not a score, a mistake count)
- LoRA adapter → a sticky note on a textbook (the book is unchanged, the note adds behaviour)
- Baseline comparison → measuring whether the medicine did anything vs. placebo

---

## Principle 6 — Make the comparison explicit

The most powerful teaching moment in this kind of experiment is the before/after or the with/without comparison. Don't make the learner infer it. State it directly.

"Fine-tuned: 99.3%. Baseline: 50.7%. The +48.6 points came from consistency and precision — not new knowledge. The base model already knew what 'cancellation' meant. It just wasn't reliable about outputting exactly one word."

The comparison tells the story that no single number can. It's also the thing the learner can point to when explaining the experiment to someone else.

---

## Principle 7 — Separate the mechanics lesson from the results lesson

Every experiment teaches two things: how to run the machinery, and what the results mean. These are different. Don't conflate them.

The mechanics lesson: how to format JSONL, what LoRA rank does, how to read a confusion matrix.

The results lesson: what 99.3% vs 50.7% tells you about when to fine-tune, what a healthy loss curve looks like, why `product_how_to` was the hardest class.

Students get confused when a session mixes these without signposting. Be explicit: "That's the mechanics. Now here's what the result actually means."

---

## Principle 8 — Calibrate to level, not to topic

The same concept needs different explanations depending on who you're talking to. The signal is not "how technical is this person" — it's "what mental models do they already have."

Someone who has shipped software but never done ML: explain in terms of software they already understand. A LoRA adapter is like a config override file that runs on top of the base model without modifying it.

Someone completely new to technical work: use everyday analogies (practice exam, sticky note).

Someone with ML background: skip the analogies, go straight to the tradeoffs.

**How to calibrate in real time:** Watch what questions they ask. If they ask "what is fine-tuning?" they need the everyday analogy. If they ask "why LoRA instead of full fine-tuning?" they already have the conceptual model and need the technical comparison.

---

## Principle 9 — End each stage with what comes next

Every stage should close with a one-line bridge to the next stage. The learner should never be left wondering "okay, what now?"

"Data is ready. Next: the LoRA config — this is where we set the knobs that control how training behaves."

This is a small thing that has a large effect on the learner feeling oriented vs. adrift.

---

## Principle 10 — The baseline comparison is always mandatory

In any experiment that compares two things (fine-tuned vs. not, with feature vs. without, new approach vs. old), the comparison run is not optional. It is the conclusion of the experiment.

Without the baseline, you don't know if you did anything. You just have a number. With the baseline, you have a claim: "this approach produced X% improvement over not doing it, for this reason."

This applies beyond ML: any A/B test, any refactor, any new tool. Always run the baseline.

---

## What this style is not

- **Not a lecture.** The learner is executing code, watching output, making decisions. You're narrating alongside them, not presenting to them.
- **Not exhaustive.** Don't explain every detail of every concept. Explain what's relevant to the current moment and the current output.
- **Not hand-holding.** Calibrate to the learner's level. Don't over-explain concepts they already have. The goal is to extend their model, not replace it.
- **Not cheerful filler.** No "great question!" or "that's a really interesting point." Just the substance.
