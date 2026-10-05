# Final pipeline

These 12 scripts produce the final submission (V27b). Run them in order from any directory; every path comes from [`config.py`](../config.py).

| # | Script | Reads | Writes | Public AUC |
|---|---|---|---|---:|
| 01 | `01_extract_yolo_kinematics.py` | test clips | `artifacts/cache_yolo_features.npz`: YOLOv8 time-to-collision and kinematics | |
| 02 | `02_solve_v9_yolo_flow.py` | test clips | `artifacts/cache_masked_flow.npz`: vehicle-masked optical flow (ego shake) | 0.714 |
| 03 | `03_solve_v16_advanced_physics.py` | test clips, V15 | `submission_v16_advanced_physics.csv`: tracking-only crash score; `submission_v17_ultimate_physics.csv`: V16 + V15 | 0.730 (V17) |
| 04 | `04_solve_v18_ultimate.py` | test clips | `artifacts/v18_cache/`: SigLIP SO400M, EVA02-L and CLIP ViT-L over 30 frames + flip TTA | 0.763 |
| 05 | `05_solve_v18_resume.py` | `v18_cache` | completes V18 with per-backbone checkpointing | |
| 06 | `06_solve_v19_semantic_velocity.py` | test clips | `artifacts/v19_cache/`: SigLIP-378 features + embedding velocity | 0.726 |
| 07 | `07_solve_v20_highres.py` | `v18_cache`, `v19_cache` | `submission_v20_highres.csv`: temporal-spike ensemble at 378 px | 0.785 |
| 08 | `08_solve_v21_transductive_pseudo.py` | V20, physics caches | `submission_v21_transductive.csv`: physics LightGBM on test pseudo-labels | 0.788 |
| 09 | `09_solve_v24_feature_blender.py` | caches + meta-submissions | `submission_v24_feature_blender.csv`: 5-fold LightGBM over 48 features | 0.789 |
| 10 | `10_solve_v25_overnight.py` | test clips, V7 | `artifacts/v25_cache/`: SigLIP2-378 and ViT-H-14 DFN5B with domain prompts; `submission_v25_standalone.csv` | 0.788 |
| 11 | `11_solve_v25b_final_blend.py` | all caches, V24, V21 | `submission_v25b_final_blend.csv`: about 100-feature LightGBM | 0.788 |
| 12 | `12_solve_v27b_physics_injection.py` | V24, V16, V25b | **`submission_v27b_physics_injection.csv`**: 85 / 8 / 7 rank blend | **0.791** |

Notes:

- **GPU stages** are 01–06 and 10. The zero-shot backbones (04–06, 10) dominate runtime, at several hours on a single consumer GPU. Each one caches per-backbone results and skips finished work on restart.
- **Meta-features.** 04/05 and 09 also read earlier submissions (V6–V20) as features. Those CSVs are committed in [`submissions/`](../submissions/), so the blenders run without re-running [`experiments/`](../experiments/).
- **Determinism.** 09 and 11 use fixed seeds. [`tools/verify_pipeline.py --full`](../tools/verify_pipeline.py) checks that 09 → 11 → 12 reproduce the submitted files exactly.
