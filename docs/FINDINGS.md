# Findings

What worked, what didn't, and the measurements behind each claim. Public leaderboard (LB) numbers are ROC-AUC on about 30% of the 1,750 test clips.

## 1. Why supervised models collapse on test

| Model | Grouped CV AUC (train) | Public LB AUC |
|---|---:|---:|
| EfficientNet-B3 + GRU, 5-fold (V1) | 0.995 | 0.538 |
| Logistic probe on CLIP + motion features | 0.991 | — |
| ResNet3D on re-labelled train (V13) | — | 0.605 |

`train_labels.csv` has 1,500 `group_id`s with **exactly two clips each, always one crash and one normal**. Per the V2 analysis, the normal clip is taken from the seconds before the crash in the same video. Test normal clips are ordinary driving from unrelated videos.

Evidence that this is what breaks transfer, rather than just a hypothesis:

- **Pair signature.** Using 48×28 grayscale thumbnails of frames 0/15/29, mutual nearest neighbours recover the true train pair with **92.6% precision** (55% of clips matched). Within those pairs, a cross-validated model's scores correlate **−0.58**, because the labels are opposite. On the 298 mutual-NN pairs in *test*, the same model's scores correlate **+0.38**: visually matched test clips mostly share a label.
- **"Sunny train, night test" is not supported.** It was an early hypothesis and appears in several docstrings, but mean frame brightness is 110.7 (train) versus 107.2 (test). Train actually has *more* dark clips: 7.4% versus 2.4% below a mean intensity of 60.

**Consequence:** train labels give a misleading training signal, and grouped CV is not a usable validation set. Everything after V6 avoids fitting train labels directly.

## 2. What moved the score

| Change | From → To | Δ LB |
|---|---|---:|
| Drop supervised training and score frames zero-shot with CLIP prompts | 0.538 → 0.677 | +0.139 |
| Add YOLO kinematics to semantic scores (V6 → V7) | 0.720 → 0.737 | +0.017 |
| Score **all 30 frames** with three backbones (SigLIP, EVA02, CLIP) and use the temporal *shape*: late spike, gradient, late-minus-early (V15 → V18) | 0.747 → 0.763 | +0.016 |
| Same temporal features at **378 px** instead of 224 px (V18 → V20) | 0.763 → 0.785 | +0.022 |
| Transductive pseudo-labels: train a physics LightGBM on V20's most confident test clips (V21) | 0.785 → 0.788 | +0.003 |
| LightGBM over all 48 temporal features + meta-scores, on pseudo-labels (V24) | 0.788 → 0.789 | +0.001 |
| Rank-inject 8% V16 physics (ρ = 0.216 with V24) + 7% V25b (V27b) | 0.789 → 0.791 | +0.002 |

The two largest gains are **not using train labels** and **resolution**. Small objects (brake lights, debris, distant vehicles) are lost at 224 px.

**Pseudo-labelling depends on teacher quality.** V5 self-trained on a 0.68 teacher and *lost* 0.039. V21 used a 0.785 teacher and gained.

## 3. What didn't work

| Idea | Result | Takeaway |
|---|---|---|
| **Semantic velocity**: rate of change of frame embeddings (V19) | 0.763 → 0.726 | Turns, bumps and lighting changes move embeddings as much as impacts do. |
| **Manifold smoothing**: k-NN diffusion of V24 scores over CLIP embeddings (V26) | 0.789 → 0.763 | Test neighbours in embedding space don't reliably share labels (see §1). |
| **Cleaning train labels** (V13, V14) | 0.605, 0.692 | Label noise wasn't the problem; train construction was. |
| **Blend everything** (V22 hand weights, V23 automatic) | 0.781, 0.785 | Most submissions correlate above 0.95; averaging them averages noise. |
| **Qwen2.5-VL-7B zero-shot**: 5 frames, 4-bit NF4, P(Yes) from next-token logits ([`kaggle/`](../kaggle/)) | AUC **0.648** on 300 labelled train clips; ρ = 0.607 with V27b | A general VLM shown 5 sparse frames is weaker than CLIP-family temporal features. It was not submitted. Engineering detail: 1.26 s per clip per T4 with one model replica per GPU. |
| **Pair-difference adjustment**: score each clip relative to its matched partner | Train: 0.694 → 0.749 (proxy model, λ = 2) | Rejected without submitting. §1 shows test pairs carry the *opposite* signature, so it would hurt on test. |

## 4. Leaderboard noise and final selection

With about 525 public clips, the sampling noise on AUC is roughly ±0.02, so V24, V25b, V27b and V28 (0.788 to 0.791) are statistically tied. Kaggle typically scores the best of the selected submissions on the private 70%. The selection therefore maximises **diversity among near-best models** instead of picking three near-copies:

| | V27b | V28 | V24 | V25b | V25 |
|---|---:|---:|---:|---:|---:|
| Public AUC | 0.7910 | 0.7902 | 0.7894 | 0.7883 | 0.7876 |
| ρ with V27b | 1.000 | 0.999 | 0.997 | 0.953 | 0.903 |
| Selected | ✅ | | | ✅ | ✅ |

V28 and V24 would add almost no independent chance. V25 is the most different model that is still competitive, and the least tuned against the public leaderboard (fixed hand weights, submitted once).

## 5. Data integrity

One exploratory submission (V12) overwrote 200 hand-annotated test clips. Hand-labelling test data is outside the competition's rules, so that script, its labels and its CSV are excluded from this repository. A code search confirms that no other script reads those labels, and none of the final submissions depend on V12, directly or through meta-features.
