# Supervised baseline (V1)

EfficientNet-B3 frame encoder with a GRU or temporal-attention head ([`models.py`](models.py)), trained on 8 sampled frames per clip with 5-fold `GroupKFold` ([`train.py`](train.py)). Inference ensembles the folds with TTA ([`predict.py`](predict.py)).

```bash
python baselines/train.py --model-type gru --backbone efficientnet_b3 --epochs 15
python baselines/predict.py
```

**Grouped CV AUC 0.995 → public LB 0.538.** This is the result that redirected the whole project; see [docs/FINDINGS.md](../docs/FINDINGS.md#1-why-supervised-models-collapse-on-test). It is kept as the reference point for every later gain.
