---
status: "done"
current_state: "All three experiments complete (Qwen LoRA, ModernBERT, SetFit). Conclusion reached: the ceiling is the data, not the model. All architectures plateau at 40–60% on natural language. No further runs planned."
next_action: "None. Experiment closed. Any future work needs a real product-support ticket corpus, not synthetic Bitext — out of scope for this kata."
things_to_know:
  - "EXPERIMENTS.md holds the full comparison table and conclusion; README.md is the public writeup."
  - "SetFit did not close the natural-language gap; more augmented data made it worse, not better."
  - "CFPB complaints (signal repo) were evaluated as a real-language source and rejected — wrong domain/taxonomy."
what_it_is: "Completed intent-classifier learning experiment: synthetic→natural generalization gap measured across a decoder LLM, an encoder, and a contrastive few-shot method."
read_next:
  - "README.md"
  - "EXPERIMENTS.md"
safe_first_action: "Read README.md, then EXPERIMENTS.md for the comparison table and conclusion."
updated_at: "2026-06-30"
updated_by: "claude"
---

## Build inbox
Free-write feature ideas, follow-ups, and "do this next" notes here. Keep coding-agent implementation detail in `SKILL.md`.
