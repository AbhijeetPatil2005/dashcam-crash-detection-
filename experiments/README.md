# Experiments

Every approach that reached the public leaderboard but is not part of the final pipeline, kept as-is so the path from 0.54 to 0.79 can be traced. Each script writes its submission to [`submissions/`](../submissions/); later blends read earlier submissions from there.

| Script | Submission | Idea | Public AUC |
|---|---|---|---:|
| [`../baselines/`](../baselines/) | V1 | Supervised EfficientNet-B3 + GRU, 5-fold GroupKFold, TTA | 0.53770 |
| `solve_v2.py` | V2 | CNN + motion + CLIP ensemble trained on train labels | 0.55737 |
| `solve_zero_shot.py` | zero-shot | Frame-difference spike, no training | 0.61681 |
| `solve_zero_shot_ensemble.py` | zero-shot | CLIP prompts + motion heuristics | 0.63884 |
| `solve_zero_shot_final.py` | zero-shot | CLIP prompts + physics heuristics | 0.67702 |
| `solve_v3_anomaly.py` | V3 | Anomaly score + zero-shot | 0.66455 |
| `solve_v4.py` | V4 | CLIP-only / ensemble | 0.69832 |
| `solve_v5_pseudo.py` | V5 | Self-training on V4 pseudo-labels | 0.64587 |
| `solve_v6_ultimate.py` | V6 | Foundation-model mega-ensemble | 0.71953 |
| `solve_v7_yolo.py` | V7 | YOLO physics + V6 semantics | 0.73669 |
| `solve_v8_video.py` | V8 | X-CLIP video foundation model | 0.72677 |
| `solve_v10_distillation.py` | V10 | CNN self-trained on V7's most confident 5% / 5% test predictions | 0.71803 |
| `solve_v11_lgb_flow.py` | V11 | LightGBM on masked flow | 0.69236 |
| `solve_v13_clean_train.py` | V13 | ResNet3D on a re-labelled train set | 0.60494 |
| `solve_v14_lgb_clean.py` | V14 | LightGBM flow on the re-labelled train set | 0.69236 |
| `solve_v15_ensemble.py` | V15 | Weighted V7 + V8 + V10 + V9 | 0.74714 |
| `solve_v22_apex.py` | V22 | Hand-weighted rank blend of V21, V20, V18, V15, V7 | 0.78103 |
| `solve_v23_autoblend.py` | V23 | Automatic blend of all submissions | 0.78455 |
| `solve_v26_manifold.py` | V26 | k-NN diffusion of V24 scores over CLIP embeddings | 0.76322 |
| `solve_v27_physics_vision_fusion.py` | V27 | LightGBM over physics + vision meta-scores (not submitted) | |
| `solve_v28_final.py` | V28 | 90% V24 + 10% V16 physics | 0.79022 |

**Patterns worth noting**

- Everything trained directly on the train labels (V1, V2, V13, V14) lands between 0.54 and 0.69. See [docs/FINDINGS.md](../docs/FINDINGS.md) for why.
- Pseudo-labelling helps only once the teacher is strong: V5 (teacher ≈ 0.68) **lost** 0.04, while V21 (teacher ≈ 0.785) **gained** 0.003.
- Hand-weighted (V22) and automatic (V23) blends of everything underperform a single well-fed LightGBM (V24). The components correlate 0.86–0.98, so there is little diversity to gain, and weaker members dilute the strong ones.

Scripts are kept verbatim apart from path handling, which now goes through [`config.py`](../config.py). Docstrings reflect the hypothesis at the time, including some that later turned out wrong (for example the "sunny train / night test" theory; see FINDINGS).
