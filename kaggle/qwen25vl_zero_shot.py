"""
Kaggle Notebook: Qwen2.5-VL-7B (4-bit) zero-shot crash scorer + V27b rank blend
===============================================================================
Setup (Kaggle notebook):
  * Accelerator: GPU T4 x2.  Internet: ON (pip + HF download), unless you attach the model.
  * Add data: competition "cooked-or-not".
  * Add data: your dataset containing submission_v27b_physics_injection.csv
  * Optional: add Kaggle Model "Qwen2.5-VL-7B-Instruct" (transformers) - auto-detected, else pulled from HF.
  * Paste this whole file into ONE cell and run.

Design notes:
  * One model replica per T4, each scoring half the clips in its own subprocess (2x throughput, no
    cross-GPU layer splitting).
  * Score = P(Yes) from the next-token logits of a single forward pass (no generate()). A free-text
    "probability" from a VLM collapses to a handful of values (0.0/0.1/0.9...) with massive ties, which is
    poison for a rank-based AUC blend; the Yes/No logit margin is continuous and ~3x faster.
  * 300 labelled train clips are scored first -> Qwen's standalone AUC is printed within minutes, so you
    can abort early if it is useless.
  * The blend weight is gated on labelled TRAIN data only: if Qwen's train AUC is weak or it ranks test
    almost identically to V27b, the script falls back to plain V27b and tells you NOT to spend the submission.
"""

import os, sys, subprocess, time, glob, json, textwrap

# ============================================================
# STEP 1: Install dependencies (parent never imports transformers; workers are fresh processes)
# ============================================================
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers==4.51.3", "accelerate>=1.2.0", "bitsandbytes>=0.45.0", "qwen-vl-utils"],
               check=False)

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

# ============================================================
# CONFIG
# ============================================================
COMP_DATA = "/kaggle/input/cooked-or-not/crash_competition_data"
if not os.path.isdir(COMP_DATA):
    COMP_DATA = sorted(glob.glob("/kaggle/input/**/crash_competition_data", recursive=True))[0]
print(f"Competition data: {COMP_DATA}")
TEST_DIR = os.path.join(COMP_DATA, "test")
TRAIN_DIR = os.path.join(COMP_DATA, "train")
V27B_PATH = "/kaggle/input/v27b-submission/submission_v27b_physics_injection.csv"
OUT_DIR = "/kaggle/working"

FRAME_INDICES = [0, 7, 14, 21, 29]
# Native clips are 384x224. Keeping the aspect ratio = no distortion and 112 vs 196 tokens per frame.
# Set to (384, 384) if you really want square inputs (slower, stretched).
RESIZE_HW = (224, 384)               # (height, width); rounded to multiples of 28 by the processor
N_VAL_TRAIN = 300                    # labelled train clips scored first for a quick AUC sanity check
DEFAULT_WEIGHT = 0.10                # user's conservative blend: 0.90*rank(V27b) + 0.10*rank(Qwen)
MIN_QWEN_AUC = 0.68                  # below this, Qwen is not blended at all
TIME_BUDGET_MIN = 100                # hard stop for scoring; leaves margin before the deadline
SEED = 42


def find_first(patterns):
    for p in patterns:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return hits[0]
    return None


def find_model_path():
    # The Kaggle Models mount is slow (~15 min/load) and its preprocessor config breaks AutoProcessor,
    # so pull the official HF snapshot once onto local disk; both workers then load from page cache.
    from huggingface_hub import snapshot_download
    t = time.time()
    path = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct", local_dir="/tmp/qwen25vl7b",
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"])
    print(f"Downloaded model to {path} in {time.time()-t:.0f}s")
    return path


V27B_PATH = V27B_PATH if os.path.exists(V27B_PATH) else find_first(
    ["/kaggle/input/**/submission_v27b_physics_injection.csv"])
assert V27B_PATH, "submission_v27b_physics_injection.csv not found under /kaggle/input"
TRAIN_LABELS = find_first([os.path.join(COMP_DATA, "train_labels_clean.csv"),
                           os.path.join(COMP_DATA, "train_labels.csv"),
                           "/kaggle/input/**/train_labels*.csv"])
MODEL_PATH = find_model_path()
print(f"V27b: {V27B_PATH}\nTrain labels: {TRAIN_LABELS}")

# ============================================================
# STEP 2: Worker script (one per GPU)
# ============================================================
WORKER_SRC = r'''
import os, sys, gc, csv, glob, time, math
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

jobs_file, out_file, ready_file, model_path, H, W = sys.argv[1:7]
H, W = int(H), int(W)
FRAME_INDICES = [0, 7, 14, 21, 29]
PROMPT = ("These are sequential frames from a dashcam video, in time order. "
          "Is there a car crash happening? Answer with only Yes or No.")

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
try:
    processor = AutoProcessor.from_pretrained(model_path)
except Exception as e:
    print(f"local processor failed ({e!r}); using hub processor", flush=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, quantization_config=bnb, torch_dtype=torch.float16,
    device_map={"": 0}, attn_implementation="sdpa", low_cpu_mem_usage=True).eval()
tok = processor.tokenizer

def single_ids(words):
    out = set()
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) == 1:
            out.add(t[0])
    return sorted(out)

YES = single_ids(["Yes", "yes", "YES", " Yes", " yes"])
NO = single_ids(["No", "no", "NO", " No", " no"])
assert YES and NO, "could not resolve Yes/No token ids"
open(ready_file, "w").write("ok")
print(f"model loaded on {torch.cuda.get_device_name(0)}; yes={YES} no={NO}", flush=True)

def load_frames(clip_dir):
    files = sorted(glob.glob(os.path.join(clip_dir, "frame_*.jpg")))
    imgs = []
    for i in FRAME_INDICES:
        p = os.path.join(clip_dir, f"frame_{i:03d}.jpg")
        if not os.path.exists(p):
            p = files[min(i, len(files) - 1)]
        imgs.append(Image.open(p).convert("RGB"))
    return imgs

def score(clip_dir):
    content = [{"type": "image", "image": im, "resized_height": H, "resized_width": W}
               for im in load_frames(clip_dir)]
    content.append({"type": "text", "text": PROMPT})
    msgs = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].float()
    margin = (torch.logsumexp(logits[YES], 0) - torch.logsumexp(logits[NO], 0)).item()
    del inputs, logits
    return margin

done = set()
if os.path.exists(out_file):
    with open(out_file) as f:
        done = {row["clip_id"] + "|" + row["split"] for row in csv.DictReader(f)}
jobs = [l.rstrip("\n").split("\t") for l in open(jobs_file) if l.strip()]
jobs = [j for j in jobs if j[1] + "|" + j[0] not in done]

new_file = not os.path.exists(out_file)
fout = open(out_file, "a", newline="")
w = csv.writer(fout)
if new_file:
    w.writerow(["split", "clip_id", "margin"])
t0 = time.time()
for n, (split, clip_id, clip_dir) in enumerate(jobs, 1):
    try:
        m = score(clip_dir)
        if not math.isfinite(m):
            m = float("nan")
    except torch.cuda.OutOfMemoryError:
        print(f"OOM on {clip_id}", flush=True)
        m = float("nan")
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"error on {clip_id}: {e!r}", flush=True)
        m = float("nan")
    w.writerow([split, clip_id, m]); fout.flush()
    if n % 50 == 0:
        gc.collect(); torch.cuda.empty_cache()
        print(f"{n}/{len(jobs)}  {(time.time()-t0)/n:.2f}s/clip  "
              f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
fout.close()
print("DONE", flush=True)
'''
worker_path = os.path.join(OUT_DIR, "qwen_worker.py")
with open(worker_path, "w") as f:
    f.write(WORKER_SRC)

# ============================================================
# STEP 3: Build jobs (labelled train subset first, then all test clips)
# ============================================================
v27b = pd.read_csv(V27B_PATH)
test_ids = v27b["clip_id"].astype(str).tolist()
assert len(test_ids) == 1750, f"expected 1750 test rows, got {len(test_ids)}"

val_df = pd.DataFrame(columns=["clip_id", "label"])
if TRAIN_LABELS and N_VAL_TRAIN > 0:
    tl = pd.read_csv(TRAIN_LABELS)
    tl = tl[tl["clip_id"].map(lambda c: os.path.isdir(os.path.join(TRAIN_DIR, str(c))))]
    per = N_VAL_TRAIN // 2
    val_df = pd.concat([g.sample(min(per, len(g)), random_state=SEED) for _, g in tl.groupby("label")])
    val_df = val_df.sample(frac=1, random_state=SEED)[["clip_id", "label"]]

jobs = [("val", str(c), os.path.join(TRAIN_DIR, str(c))) for c in val_df["clip_id"]]
jobs += [("test", c, os.path.join(TEST_DIR, c)) for c in test_ids]

n_gpu = max(1, torch.cuda.device_count())
print(f"{len(jobs)} clips ({len(val_df)} val + {len(test_ids)} test) on {n_gpu} GPU(s)")
out_files = []
for g in range(n_gpu):
    jf = os.path.join(OUT_DIR, f"jobs_gpu{g}.tsv")
    with open(jf, "w") as f:
        for j in jobs[g::n_gpu]:   # interleaved -> both GPUs finish val early
            f.write("\t".join(j) + "\n")
    out_files.append(os.path.join(OUT_DIR, f"qwen_scores_gpu{g}.csv"))

# ============================================================
# STEP 4: Launch one worker per GPU (4-bit loading streams shard by shard, so both fit in host RAM)
# ============================================================
procs, logs = [], []
start = time.time()
for g in range(n_gpu):
    ready = os.path.join(OUT_DIR, f"ready_gpu{g}")
    if os.path.exists(ready):
        os.remove(ready)
    log_path = os.path.join(OUT_DIR, f"worker_gpu{g}.log")
    logs.append(log_path)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1")
    procs.append(subprocess.Popen(
        [sys.executable, worker_path, os.path.join(OUT_DIR, f"jobs_gpu{g}.tsv"), out_files[g], ready,
         MODEL_PATH, str(RESIZE_HW[0]), str(RESIZE_HW[1])],
        stdout=open(log_path, "a"), stderr=subprocess.STDOUT, env=env))
for g, p in enumerate(procs):
    ready = os.path.join(OUT_DIR, f"ready_gpu{g}")
    while not os.path.exists(ready) and p.poll() is None:
        time.sleep(10)
    if p.poll() is not None:
        print(open(logs[g]).read()[-3000:])
        raise RuntimeError(f"worker {g} died while loading the model - see log above")
    print(f"GPU{g} worker ready after {time.time()-start:.0f}s")


def read_scores():
    frames = []
    for f in out_files:
        try:
            frames.append(pd.read_csv(f, on_bad_lines="skip"))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(columns=["split", "clip_id", "margin"])
    return pd.concat(frames).drop_duplicates(["split", "clip_id"], keep="last")


val_reported = False
while any(p.poll() is None for p in procs):
    time.sleep(60)
    s = read_scores()
    nv, nt = (s["split"] == "val").sum(), (s["split"] == "test").sum()
    el = (time.time() - start) / 60
    print(f"[{el:5.1f} min] val {nv}/{len(val_df)}  test {nt}/{len(test_ids)}  "
          f"nan={s['margin'].isna().sum()}")
    if not val_reported and len(val_df) and nv >= len(val_df):
        v = val_df.merge(s[s["split"] == "val"], on="clip_id").dropna()
        print(f"\n>>> Qwen zero-shot AUC on {len(v)} labelled train clips: "
              f"{roc_auc_score(v['label'], v['margin']):.4f}  (< {MIN_QWEN_AUC} => will not be blended)\n")
        val_reported = True
    if el > TIME_BUDGET_MIN:
        print("Time budget hit - stopping workers, blending with what we have")
        for p in procs:
            p.terminate()
        break
for g, lp in enumerate(logs):
    print(f"--- worker {g} log tail ---\n" + "".join(open(lp).readlines()[-5:]))

# ============================================================
# STEP 5: Evaluate, gate, blend
# ============================================================
s = read_scores()
val = val_df.merge(s[s["split"] == "val"], on="clip_id").dropna()
qwen_auc = roc_auc_score(val["label"], val["margin"]) if len(val) > 20 else float("nan")

qt = s[s["split"] == "test"].set_index("clip_id")["margin"]
qwen = v27b[["clip_id"]].copy()
qwen["margin"] = qwen["clip_id"].map(qt)
coverage = qwen["margin"].notna().mean()
qwen["margin"] = qwen["margin"].fillna(qwen["margin"].median())
qwen["label"] = 1.0 / (1.0 + np.exp(-qwen["margin"]))
qwen[["clip_id", "label"]].to_csv(os.path.join(OUT_DIR, "submission_qwen_standalone.csv"), index=False)

r_v27b = rankdata(v27b["label"].values) / len(v27b)
r_qwen = rankdata(qwen["margin"].values) / len(qwen)
rho = spearmanr(r_v27b, r_qwen).correlation
print(f"\nQwen train-subset AUC: {qwen_auc:.4f} | test coverage: {coverage:.1%} | "
      f"Spearman(V27b, Qwen) on test: {rho:.3f}")


def blend(w):
    return (1 - w) * r_v27b + w * r_qwen


weight = DEFAULT_WEIGHT
if not (qwen_auc >= MIN_QWEN_AUC) or coverage < 0.95:
    weight = 0.0
    print("Qwen is too weak / incomplete -> weight 0")
elif rho > 0.90:
    weight = 0.0
    print("Qwen ranks almost identically to V27b -> nothing to add, weight 0")

final = v27b[["clip_id"]].copy()
final["label"] = blend(weight)
assert final["label"].notna().all() and len(final) == 1750
final.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)
print(f"\nFINAL: {1-weight:.2f}*rank(V27b) + {weight:.2f}*rank(Qwen) -> /kaggle/working/submission.csv")
if weight == 0.0:
    print("!!! submission.csv is just V27b re-ranked (same LB score). Do NOT spend your last submission on it.")
