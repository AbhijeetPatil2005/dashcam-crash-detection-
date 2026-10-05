"""
Reproducibility check for the submitted CSVs.

Default (no GPU, no competition data, ~1 s):
    Re-runs the final blend (pipeline/12) on a temporary copy of the committed component submissions
    and checks the result matches the committed V27b file exactly.

--full (needs feature caches in CRASH_CACHE_DIR):
    Also re-runs the LightGBM blenders V24 and V25b from cached features before the final blend.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import SUBMISSIONS_DIR

FINAL = "submission_v27b_physics_injection.csv"
STAGES = {
    "09_solve_v24_feature_blender.py": "submission_v24_feature_blender.csv",
    "11_solve_v25b_final_blend.py": "submission_v25b_final_blend.csv",
    "12_solve_v27b_physics_injection.py": FINAL,
}


def compare(name, produced, reference):
    a, b = pd.read_csv(produced), pd.read_csv(reference)
    same_ids = len(a) == len(b) and (a["clip_id"].values == b["clip_id"].values).all()
    diff = float(np.abs(a["label"].values - b["label"].values).max()) if same_ids else float("inf")
    status = "OK  " if same_ids and diff < 1e-9 else "FAIL"
    print(f"  [{status}] {name:<42} rows={len(a)}  max|diff|={diff:.2e}")
    return status == "OK  "


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="also re-run the V24 and V25b blenders")
    args = parser.parse_args()
    scripts = list(STAGES) if args.full else ["12_solve_v27b_physics_injection.py"]

    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "submissions")
        shutil.copytree(SUBMISSIONS_DIR, work)
        env = dict(os.environ, CRASH_SUBMISSIONS_DIR=work)
        if not os.environ.get("CRASH_DATA_DIR"):
            # The blend only needs the test clip ids; take them from the committed submission.
            data = os.path.join(tmp, "data")
            os.makedirs(data)
            ids = pd.read_csv(os.path.join(SUBMISSIONS_DIR, FINAL))[["clip_id"]].assign(label=0.5)
            ids.to_csv(os.path.join(data, "sample_submission.csv"), index=False)
            env["CRASH_DATA_DIR"] = data

        ok = True
        print("Reproducing submitted files:")
        for script in scripts:
            subprocess.run([sys.executable, str(ROOT / "pipeline" / script)], env=env, check=True,
                           stdout=subprocess.DEVNULL)
            out = STAGES[script]
            ok &= compare(out, os.path.join(work, out), os.path.join(SUBMISSIONS_DIR, out))

    print("All submissions reproduced exactly." if ok else "Mismatch - see above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
