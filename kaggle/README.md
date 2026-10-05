# Qwen2.5-VL zero-shot (Kaggle notebook)

[`qwen25vl_zero_shot.py`](qwen25vl_zero_shot.py) is a single-cell Kaggle script that scores every clip with **Qwen2.5-VL-7B-Instruct** and rank-blends the result into V27b, but only if the model earns it.

**Engineering**

- **4-bit NF4** (bitsandbytes, double quantisation) brings the 7B model to about 7.4 GB peak VRAM, so it fits a T4.
- **Data-parallel across both T4s:** one model replica per GPU in its own subprocess, each scoring half the clips. This avoids the slow cross-GPU layer split of `device_map="auto"`.
- **Logit scoring instead of generation:** a single forward pass, with `P(Yes) = softmax(logits[Yes-tokens] vs logits[No-tokens])`. Asking the model to *write* a probability collapses to a few values (0.0 / 0.1 / 0.9) with massive ties, which wrecks a rank-based AUC.
- **Native aspect ratio:** frames 0/7/14/21/29 at 384×224 (112 visual tokens each) rather than stretching to 384×384.
- **Fault-tolerant:** per-clip error handling, OOM recovery, results flushed per clip (resumable), cache clearing every 50 clips, and a hard time budget.
- **Self-gating:** it first scores 300 labelled train clips. Qwen is blended only if its train AUC is at least 0.68 and its test rank correlation with V27b is at most 0.90; otherwise the script writes V27b unchanged.

**Result:** 2,050 clips in about 22 minutes at 1.26 s per clip per GPU. Train-subset AUC was **0.648**, so the gate set the blend weight to 0 and this was not submitted. See [docs/FINDINGS.md](../docs/FINDINGS.md#3-what-didnt-work).

**Run:** attach the competition data and a dataset containing `submission_v27b_physics_injection.csv`, choose GPU T4 ×2 with internet on, and paste the file into one cell.
