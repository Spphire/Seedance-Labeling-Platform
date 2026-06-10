from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(os.environ.get("SEEDANCE_PLATFORM_ROOT", Path(__file__).resolve().parents[2])).resolve()
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
DB_PATH = ROOT / "db.sqlite3"

EPISODES_DIR = DATA_DIR / "episodes"
HEAD_VIDEOS_DIR = DATA_DIR / "head_videos"
CLIPS_DIR = DATA_DIR / "clips"
GENERATED_DIR = DATA_DIR / "generated"
ACCEPTED_DIR = DATA_DIR / "accepted_clips"
ARCHIVED_ANCHORS_DIR = DATA_DIR / "archived_anchor_candidates"
FINAL_DIR = DATA_DIR / "final_episodes"
FINAL_DATASET_DIR = DATA_DIR / "final_dataset"
LOGS_DIR = ROOT / "logs"
REFERENCE_IMAGES_DIR = ROOT / "app" / "reference_images"


def ensure_dirs() -> None:
    for path in [
        CONFIG_DIR,
        EPISODES_DIR,
        HEAD_VIDEOS_DIR,
        CLIPS_DIR,
        GENERATED_DIR,
        ACCEPTED_DIR,
        ARCHIVED_ANCHORS_DIR,
        FINAL_DIR,
        FINAL_DATASET_DIR,
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
