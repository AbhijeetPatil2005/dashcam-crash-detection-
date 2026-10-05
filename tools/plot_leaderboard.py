"""Render the public-leaderboard progression chart used in the README (light + dark variants)."""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPO_ROOT, SUBMISSIONS_DIR

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#a8a7a1", grid="#e6e5e0", accent="#2a78d6"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#6b6a64", grid="#2e2e2c", accent="#3987e5"),
}
FINAL_PICKS = {
    "submission_v27b_physics_injection.csv": "V27b  0.7910",
    "submission_v25b_final_blend.csv": "V25b  0.7883",
    "submission_v25_standalone.csv": "V25  0.7876",
}
MILESTONES = {
    "submission_v1_effnet_gru.csv": "Supervised EffNet-B3 + GRU",
    "submission_zero_shot_final.csv": "Zero-shot CLIP + physics",
    "submission_v7_yolo.csv": "YOLO kinematics",
}


def render(theme, out_path):
    t = THEMES[theme]
    df = pd.read_csv(os.path.join(SUBMISSIONS_DIR, "leaderboard.csv"), parse_dates=["submitted_utc"])
    df = df.sort_values("submitted_utc")
    df["best"] = df["public_auc"].cummax()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    ax.scatter(df["submitted_utc"], df["public_auc"], s=26, color=t["muted"], zorder=2,
               edgecolor=t["surface"], linewidth=1.5, label="Submission")
    ax.step(df["submitted_utc"], df["best"], where="post", color=t["accent"], linewidth=2, zorder=3,
            label="Best so far")

    picks = df[df["file"].isin(FINAL_PICKS)]
    ax.scatter(picks["submitted_utc"], picks["public_auc"], s=60, color=t["accent"], zorder=4,
               edgecolor=t["surface"], linewidth=2, label="Final selection")
    label_x = pd.Timestamp(df["submitted_utc"].max()) - pd.Timedelta(hours=77)
    label_y = {"submission_v27b_physics_injection.csv": 0.805, "submission_v25b_final_blend.csv": 0.785,
               "submission_v25_standalone.csv": 0.765}
    for _, r in picks.iterrows():
        ax.annotate(FINAL_PICKS[r["file"]], (r["submitted_utc"], r["public_auc"]),
                    xytext=(label_x, label_y[r["file"]]), textcoords="data", ha="right", va="center",
                    color=t["ink"], fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=t["ink2"], linewidth=0.8, shrinkA=4, shrinkB=5))
    for _, r in df[df["file"].isin(MILESTONES)].iterrows():
        ax.annotate(MILESTONES[r["file"]], (r["submitted_utc"], r["public_auc"]), xytext=(8, -4),
                    textcoords="offset points", ha="left", va="top", color=t["ink2"], fontsize=8.5)

    ax.set_ylim(0.5, 0.82)
    ax.set_ylabel("Public leaderboard ROC-AUC", color=t["ink2"])
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], length=0)
    ax.set_title("32 submissions in 13 days: 0.538 → 0.791", loc="left", color=t["ink"], fontsize=12,
                 fontweight="bold", pad=12)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    fig.tight_layout()
    fig.savefig(out_path, facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    out_dir = os.path.join(REPO_ROOT, "assets")
    os.makedirs(out_dir, exist_ok=True)
    for theme in THEMES:
        render(theme, os.path.join(out_dir, f"leaderboard_{theme}.png"))
    print(f"Saved charts to {out_dir}")
