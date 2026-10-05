# Tools

| Script | Purpose |
|---|---|
| `verify_pipeline.py` | Rebuilds the submitted CSVs and checks for an exact match (`--full` re-runs the LightGBM blenders). |
| `plot_leaderboard.py` | Renders `assets/leaderboard_{light,dark}.png` from `submissions/leaderboard.csv`. |
| `score_train_set.py` | Scores every train clip with CLIP to flag likely mislabelled clips. |
| `train_labeler.py` | OpenCV review tool for the clips flagged above, used to build the re-labelled train set for V13/V14. |
| `check_leak.py` | Checks for near-duplicate frames between splits. |
| `find_duplicates.py` | Finds exact-duplicate test clips by frame hash. |
