# SKILL.md — intent_classifier

## Current phase: Stage 2 complete — ready to train

## Stages

- [x] Stage 1 — Data prep (`scripts/prepare_data.py`) — DONE
  - 720 examples, 8 balanced classes (100 each), 576 train / 144 val
  - Output: `data/train.jsonl`, `data/val.jsonl`
- [x] Stage 2 — Config (`config/lora_config.yaml`) — DONE
  - LoRA rank 8, alpha 16, lr 1e-4, 5 epochs, batch 4
- [x] Stage 3 prep — Eval script (`scripts/eval.py`) — DONE (runs after training)
- [x] Stage 3 — Training — DONE (loss 1.282 → 0.151, 720 iters, ~2GB peak memory)
- [x] Stage 4 — Eval fine-tuned model — DONE (99.3% accuracy, 143/144 val examples)
- [x] Stage 5 — Baseline comparison — DONE (50.7% baseline; +48.6pp gain from fine-tuning)

## Training command (when ready)

```bash
mlx_lm.lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train \
  --data data/ \
  --lora-layers 8 \
  --batch-size 4 \
  --num-epochs 5 \
  --learning-rate 1e-4 \
  --adapter-path adapters/ \
  --val-batches 10
```

## Notes

- Bitext has `{{Order Number}}` template placeholders in messages — synthetic but fine for mechanics learning
- bug_report class was thin in Bitext (only 20 examples) — patched with 20 manual examples in prepare_data.py
- HF_TOKEN warning on download is cosmetic — dataset is public, no auth needed
