"""
Central path configuration. Every script imports its paths from here, so the repo
can be cloned anywhere and pointed at the data with environment variables.

    CRASH_DATA_DIR         competition data (train/, test/, *.csv)  default: data/crash_competition_data
    CRASH_CACHE_DIR        extracted features / embeddings           default: artifacts/
    CRASH_SUBMISSIONS_DIR  submission CSVs (read by the blenders)    default: submissions/
"""
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get("CRASH_DATA_DIR", os.path.join(REPO_ROOT, "data", "crash_competition_data"))
CACHE_DIR = os.environ.get("CRASH_CACHE_DIR", os.path.join(REPO_ROOT, "artifacts"))
SUBMISSIONS_DIR = os.environ.get("CRASH_SUBMISSIONS_DIR", os.path.join(REPO_ROOT, "submissions"))

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
