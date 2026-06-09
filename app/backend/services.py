from __future__ import annotations

import json
import math
import secrets
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from . import db
from .ids import parse_uuids
from .locks import LockError, active_lock, locks_by_resource, require_lock, require_no_active_lock
from .nedf import extract_head_video, fetch_episode
from .paths import ACCEPTED_DIR, ARCHIVED_ANCHORS_DIR, CLIPS_DIR, EPISODES_DIR, FINAL_DIR, GENERATED_DIR, HEAD_VIDEOS_DIR
from .settings import (
    COLLECTOR_ONLY_PRESET_ID,
    DEFAULT_GENERATION_PRESET_ID,
    IPHONE2DEPLOY_PRESET_ID,
    load_settings,
    public_url_for,
    seedance_api_key_pool,
)
from .video import (
    black_video,
    clip_plan,
    concat_videos_precise,
    compose_continuity_input,
    cut_clip,
    normalize_accepted,
    requested_seedance_duration,
    reverse_video,
    stitch_videos,
    transcode_760x570,
    trim_video,
    video_duration,
)
from ..seedance.client import SeedanceClient


_STITCH_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="seedance-stitch")
_STITCH_LOCK = threading.Lock()
_STITCHING_EPISODES: set[str] = set()
_PREVIEW_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="seedance-preview")
_PREVIEW_LOCK = threading.Lock()
_PREVIEWING_EPISODES: set[str] = set()
STITCH_LOCK_OWNER_ID = "system-stitcher"
STITCH_LOCK_OWNER_NAME = "系统合成"
STITCH_LOCK_TTL_SEC = 60 * 60 * 24
_GENERATION_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="seedance-generation")
_GENERATION_CONDITION = threading.Condition()
_GENERATION_ACTIVE = 0
_SEEDANCE_KEY_ACTIVE: dict[str, int] = {}
_GENERATION_WORKER_LOCK = threading.Lock()
_GENERATION_WORKER_JOB_IDS: set[int] = set()
_GENERATION_WATCHDOG_STARTED = False
GENERATION_CANDIDATE_STATUSES = ("pending", "generated_failed", "rejected")
GENERATION_CANDIDATE_STATUS_PLACEHOLDERS = ",".join("?" for _ in GENERATION_CANDIDATE_STATUSES)
GENERATION_WATCHDOG_INTERVAL_SEC = 30
GENERATION_STALE_RUNNING_SEC = 180
GENERATION_DOWNLOAD_RECOVERY_LIMIT = 3
DEFAULT_REQUEST_MODE = "mock"
ANCHOR_CLIP_DURATION_SEC = 4
ANCHOR_STAGE_REPLACE_ARM = "replace_arm"
ANCHOR_STAGE_REPLACE_COLLECTOR = "replace_collector"
ANCHOR_STAGE_OFFICIAL = "official"
CONTINUITY_INPUT_KINDS = {"anchor", "rolling"}
CONTINUITY_DIRECTIONS = {"anchor", "forward", "backward"}
MIN_SEEDANCE_INPUT_SEC = 4
MAX_SEEDANCE_INPUT_SEC = 15
TIME_EPSILON = 1e-3


def submit_episodes(text: str) -> list[dict[str, Any]]:
    settings = load_settings()
    uuids = parse_uuids(text)
    now = db.now()
    with db.connect() as conn:
        for uuid in uuids:
            remote_path = f"{settings['dm3_host']}:{settings['dm3_nedf_root'].rstrip('/')}/{uuid}"
            local_path = str((EPISODES_DIR / uuid).resolve())
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', ?, ?)
                ON CONFLICT(uuid) DO UPDATE SET remote_path=excluded.remote_path, updated_at=excluded.updated_at
                """,
                (uuid, remote_path, local_path, now, now),
            )
    return list_episodes()


def submit_and_preprocess_episodes(
    text: str,
    fetch_remote: bool = True,
    lock_tokens: dict[str, str] | None = None,
) -> dict[str, Any]:
    episodes = submit_episodes(text)
    uuids = parse_uuids(text)
    if not uuids:
        return {"episodes": episodes, "preprocess": []}
    return {"episodes": episodes, "preprocess": preprocess(uuids, fetch_remote, lock_tokens)}


def list_episodes() -> list[dict[str, Any]]:
    episodes = db.rows("SELECT * FROM episodes ORDER BY created_at DESC")
    episode_locks = locks_by_resource("episode")
    counts = {
        row["episode_uuid"]: row
        for row in db.rows(
            """
            SELECT
                episode_uuid,
                COUNT(*) AS clip_count,
                SUM(CASE WHEN status='accepted' THEN 1 ELSE 0 END) AS accepted_clip_count,
                SUM(CASE WHEN status='generated' THEN 1 ELSE 0 END) AS generated_clip_count,
                SUM(CASE WHEN status='generating' THEN 1 ELSE 0 END) AS generating_clip_count,
                SUM(CASE WHEN status='generated_failed' THEN 1 ELSE 0 END) AS generated_failed_clip_count,
                SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending_clip_count,
                SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END) AS rejected_clip_count,
                SUM(CASE WHEN status='flagged' THEN 1 ELSE 0 END) AS flagged_clip_count
            FROM clips
            GROUP BY episode_uuid
            """
        )
    }
    clip_paths_by_episode: dict[str, list[str]] = {}
    clips_by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in db.rows("SELECT * FROM clips"):
        clip_paths_by_episode.setdefault(row["episode_uuid"], []).append(row["local_path"])
        clips_by_episode.setdefault(row["episode_uuid"], []).append(row)
    for episode in episodes:
        aggregate = counts.get(episode["uuid"], {})
        clip_count = int(aggregate.get("clip_count") or 0)
        accepted = int(aggregate.get("accepted_clip_count") or 0)
        generated = int(aggregate.get("generated_clip_count") or 0)
        flagged = int(aggregate.get("flagged_clip_count") or 0)
        rejected = int(aggregate.get("rejected_clip_count") or 0)
        for key in [
            "clip_count",
            "accepted_clip_count",
            "generated_clip_count",
            "generating_clip_count",
            "generated_failed_clip_count",
            "pending_clip_count",
            "rejected_clip_count",
            "flagged_clip_count",
        ]:
            episode[key] = int(aggregate.get(key) or 0)
        health, health_reason = episode_preprocess_health(episode, clip_paths_by_episode.get(episode["uuid"], []))
        episode["preprocess_health"] = health
        episode["preprocess_health_reason"] = health_reason
        episode.update(rolling_episode_progress(episode, clips_by_episode.get(episode["uuid"], [])))
        episode["review_remaining_count"] = max(clip_count - accepted, 0)
        episode["manual_decision_count"] = generated + flagged + rejected
        episode["episode_stage"] = describe_episode_stage(episode)
        episode["lock"] = episode_locks.get(episode["uuid"])
        head_path = episode.get("head_video_path")
        episode["head_video_url"] = static_url_from_path(head_path, HEAD_VIDEOS_DIR, "head_videos") if head_path else None
        final_path = episode.get("final_video_path")
        episode["final_url"] = static_url_from_path(final_path, FINAL_DIR, "final") if final_path else None
        preview_path = episode.get("preview_video_path")
        episode["preview_url"] = static_url_from_path(preview_path, FINAL_DIR, "final") if preview_path else None
    return episodes


def episode_preprocess_health(episode: dict[str, Any], clip_paths: list[str]) -> tuple[str, str]:
    if episode.get("status") != "preprocessed":
        return "not_ready", ""
    head_value = episode.get("head_video_path")
    if not head_value:
        return "damaged", "head video path missing"
    if not Path(head_value).exists():
        return "damaged", "head video file missing"
    missing = sum(1 for path in clip_paths if not path or not Path(path).exists())
    if missing:
        return "damaged", f"{missing} clip file(s) missing"
    return "ok", ""


def describe_episode_stage(episode: dict[str, Any]) -> str:
    clip_count = int(episode.get("clip_count") or 0)
    accepted = int(episode.get("accepted_clip_count") or 0)
    final_status = episode.get("final_status") or "missing"
    continuity_state = episode.get("continuity_state") or ""
    if episode.get("preprocess_health") == "damaged":
        return "预处理文件疑似损坏，需要重新预处理"
    if continuity_state == "prepare_head":
        return "等待导入并准备 head"
    if continuity_state == "select_anchor":
        return "head 已就绪，等待选择锚点候选"
    if continuity_state == "anchor_candidates":
        if int(episode.get("generating_clip_count") or 0):
            return "锚点候选生成中"
        if int(episode.get("generated_failed_clip_count") or 0):
            return "锚点候选生成失败，建议重跑"
        if int(episode.get("generated_clip_count") or 0) or int(episode.get("flagged_clip_count") or 0):
            return "请选择一个锚点候选保留"
        return "锚点候选待生成"
    if continuity_state == "bidirectional":
        if int(episode.get("generating_clip_count") or 0):
            return "连续生成中"
        if int(episode.get("generated_failed_clip_count") or 0):
            return "连续生成失败，建议重跑"
        if int(episode.get("generated_clip_count") or 0) or int(episode.get("flagged_clip_count") or 0):
            return "连续片段待审核"
        if int(episode.get("pending_clip_count") or 0) or int(episode.get("rejected_clip_count") or 0):
            return "连续片段待生成"
        return "双向滚动推进中"
    if continuity_state == "ready_to_stitch":
        return "时间线已覆盖，等待合成 final"
    if continuity_state == "stitching":
        return "时间线已覆盖，正在合成 final"
    if continuity_state == "complete":
        return "时间线已覆盖，final 已合成"
    if clip_count == 0 and episode.get("status") == "preprocessed":
        return "head 已就绪，等待选择锚点候选"
    if clip_count == 0:
        return "未切片"
    if accepted == clip_count:
        if final_status == "ready":
            return "全部保留，final 已合成"
        if final_status == "stitching":
            return "全部保留，正在合成 final"
        if final_status == "failed":
            return "全部保留，final 合成失败"
        return "全部保留，等待合成 final"
    if int(episode.get("generating_clip_count") or 0):
        return "生成中"
    if int(episode.get("generated_failed_clip_count") or 0):
        return "有生成失败，建议重跑"
    if int(episode.get("manual_decision_count") or 0):
        return "待人工审核或重跑"
    if int(episode.get("pending_clip_count") or 0):
        return "待生成"
    return "处理中"


def rolling_episode_progress(episode: dict[str, Any], clips: list[dict[str, Any]]) -> dict[str, Any]:
    summary = continuity_timeline_summary(episode["uuid"], clips, episode)
    rolling_clips = [clip for clip in clips if clip.get("input_kind") == "rolling"]
    accepted_sec = float(summary.get("accepted_sec") or 0.0)
    planned_sec = summary.get("total_sec")
    complete = bool(summary.get("complete"))
    return {
        "continuity_state": summary.get("state") or episode.get("continuity_state") or "select_anchor",
        "continuity_anchor_clip_id": summary.get("anchor_clip_id"),
        "continuity_anchor_candidate_count": summary.get("anchor_candidate_count"),
        "continuity_coverage_start_sec": summary.get("coverage_start_sec"),
        "continuity_coverage_end_sec": summary.get("coverage_end_sec"),
        "continuity_accepted_sec": accepted_sec,
        "continuity_total_sec": planned_sec,
        "continuity_complete": complete,
        "rolling_clip_count": len(rolling_clips),
        "legacy_clip_count": len([clip for clip in clips if clip.get("input_kind") not in CONTINUITY_INPUT_KINDS]),
        "rolling_planned_sec": planned_sec,
        "rolling_planned_clip_count": None,
        "rolling_accepted_sec": accepted_sec,
        "rolling_remaining_sec": max(0.0, float(planned_sec) - accepted_sec) if planned_sec is not None else None,
        "rolling_complete": complete,
        "rolling_plan_error": summary.get("error") or "",
    }


def list_clips() -> list[dict[str, Any]]:
    clips = db.rows(
        """
        SELECT c.*, e.final_status
        FROM clips c
        JOIN episodes e ON e.uuid = c.episode_uuid
        ORDER BY c.episode_uuid,
                 COALESCE(c.timeline_start_sec, c.source_start_sec, c.start_sec, 0),
                 c.clip_index
        """
    )
    clip_locks = locks_by_resource("clip")
    latest_jobs = {
        row["clip_id"]: row
        for row in db.rows(
            """
            SELECT j.*
            FROM generation_jobs j
            JOIN (
                SELECT clip_id, MAX(created_at) AS max_created_at
                FROM generation_jobs
                WHERE status IN ('running','succeeded','failed')
                GROUP BY clip_id
            ) latest ON latest.clip_id = j.clip_id AND latest.max_created_at = j.created_at
            """
        )
    }
    for clip in clips:
        input_video_url = clip_input_video_url(clip)
        raw_video_url = raw_clip_video_url(clip)
        clip["input_video_url"] = input_video_url
        clip["raw_video_url"] = raw_video_url
        clip["video_url"] = clip_display_video_url(clip, input_video_url, raw_video_url)
        latest_job = latest_jobs.get(clip["id"])
        clip["latest_job_id"] = latest_job["id"] if latest_job else None
        clip["lock"] = clip_locks.get(str(clip["id"]))
        clip["latest_job"] = enrich_job_timing(latest_job) if latest_job else None
        clip["generated_url"] = (
            static_url_from_path(latest_job.get("output_path"), GENERATED_DIR, "generated")
            if latest_job and latest_job.get("status") == "succeeded"
            else None
        )
    return clips


def list_jobs() -> list[dict[str, Any]]:
    jobs = db.rows(
        """
        SELECT j.*, c.episode_uuid, c.clip_index, c.duration_sec, c.local_path AS clip_path
        FROM generation_jobs j
        JOIN clips c ON c.id = j.clip_id
        ORDER BY j.created_at DESC
        """
    )
    for job in jobs:
        job["generated_url"] = static_url_from_path(job.get("output_path"), GENERATED_DIR, "generated")
        enrich_job_timing(job)
    return jobs


def list_seedance_usage(limit: int = 100) -> dict[str, Any]:
    calls = db.rows(
        """
        SELECT
            a.*,
            c.episode_uuid,
            c.clip_index
        FROM seedance_api_calls a
        LEFT JOIN clips c ON c.id = a.clip_id
        ORDER BY a.created_at DESC
        LIMIT ?
        """,
        (max(1, int(limit)),),
    )
    summary_rows = db.rows(
        """
        SELECT
            COALESCE(operator_id, '') AS operator_id,
            COALESCE(operator_name, '') AS operator_name,
            COUNT(*) AS call_count,
            SUM(CASE WHEN status='succeeded' THEN 1 ELSE 0 END) AS succeeded_count,
            SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed_count,
            SUM(COALESCE(requested_duration_sec, 0)) AS requested_duration_sec,
            COUNT(DISTINCT clip_id) AS clip_count,
            MAX(created_at) AS last_call_at
        FROM seedance_api_calls
        GROUP BY COALESCE(operator_id, ''), COALESCE(operator_name, '')
        ORDER BY last_call_at DESC
        """
    )
    key_summary_rows = db.rows(
        """
        SELECT
            COALESCE(api_key_id, '') AS api_key_id,
            COALESCE(api_key_name, '') AS api_key_name,
            COUNT(*) AS call_count,
            SUM(CASE WHEN status='succeeded' THEN 1 ELSE 0 END) AS succeeded_count,
            SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed_count,
            SUM(COALESCE(requested_duration_sec, 0)) AS requested_duration_sec,
            COUNT(DISTINCT clip_id) AS clip_count,
            MAX(created_at) AS last_call_at
        FROM seedance_api_calls
        GROUP BY COALESCE(api_key_id, ''), COALESCE(api_key_name, '')
        ORDER BY last_call_at DESC
        """
    )
    for call in calls:
        call["usage"] = parse_json_text(call.get("usage_json"))
        call.pop("usage_json", None)
        call.pop("raw_response_json", None)
    return {"summary": summary_rows, "key_summary": key_summary_rows, "recent_calls": calls}


def list_reviewer_activity(limit: int = 100) -> dict[str, Any]:
    reviews = db.rows(
        """
        SELECT
            r.*,
            c.episode_uuid,
            c.clip_index,
            c.status AS clip_status,
            j.mode AS job_mode,
            j.task_id
        FROM reviews r
        LEFT JOIN clips c ON c.id = r.clip_id
        LEFT JOIN generation_jobs j ON j.id = r.job_id
        ORDER BY r.reviewed_at DESC
        LIMIT ?
        """,
        (max(1, int(limit)),),
    )
    summary_rows = db.rows(
        """
        SELECT
            COALESCE(operator_id, '') AS operator_id,
            COALESCE(operator_name, '') AS operator_name,
            COUNT(*) AS review_count,
            SUM(CASE WHEN decision='accept' THEN 1 ELSE 0 END) AS accept_count,
            SUM(CASE WHEN decision='reject' THEN 1 ELSE 0 END) AS reject_count,
            SUM(CASE WHEN decision='flag' THEN 1 ELSE 0 END) AS flag_count,
            SUM(CASE WHEN decision='rerun' THEN 1 ELSE 0 END) AS rerun_count,
            COUNT(DISTINCT clip_id) AS clip_count,
            MAX(reviewed_at) AS last_reviewed_at
        FROM reviews
        GROUP BY COALESCE(operator_id, ''), COALESCE(operator_name, '')
        ORDER BY last_reviewed_at DESC
        """
    )
    return {"summary": summary_rows, "recent_reviews": reviews}


def enrich_job_timing(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if not job:
        return job
    now = db.now()
    started = job.get("started_at")
    completed = job.get("completed_at")
    elapsed = None
    if started:
        elapsed = max(0.0, float((completed or now) - started))
    estimate = job.get("estimated_total_sec")
    job["elapsed_sec"] = elapsed
    job["remaining_estimated_sec"] = (
        max(0.0, float(estimate) - float(elapsed))
        if estimate is not None and elapsed is not None and job.get("status") == "running"
        else None
    )
    job["progress_pct"] = (
        max(1, min(99, int((float(elapsed) / float(estimate)) * 100)))
        if estimate and elapsed is not None and job.get("status") == "running"
        else (100 if job.get("status") == "succeeded" else 0)
    )
    job["seconds_per_video_second"] = (
        float(elapsed) / float(job["requested_duration_sec"])
        if elapsed is not None and job.get("requested_duration_sec")
        else None
    )
    return job


def refresh_clip_public_urls() -> int:
    updated = 0
    now = db.now()
    clips = db.rows("SELECT id, episode_uuid, local_path, public_url FROM clips")
    with db.connect() as conn:
        for clip in clips:
            try:
                rel = Path(clip["local_path"]).resolve().relative_to(CLIPS_DIR.resolve())
            except ValueError:
                rel = Path(clip["episode_uuid"]) / Path(clip["local_path"]).name
            public_url = public_url_for("clips", rel)
            if public_url != clip["public_url"]:
                conn.execute(
                    "UPDATE clips SET public_url=?, updated_at=? WHERE id=?",
                    (public_url, now, clip["id"]),
                )
                updated += 1
    return updated


def static_url_from_path(path_value: str | None, root: Path, prefix: str) -> str | None:
    if not path_value:
        return None
    try:
        rel = Path(path_value).resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return f"/{prefix}/{rel.as_posix()}"


def static_url_from_clip_input_path(path_value: str | None) -> str | None:
    return static_url_from_path(path_value, CLIPS_DIR, "clips") or static_url_from_path(path_value, ACCEPTED_DIR, "accepted")


def raw_clip_path(clip: dict[str, Any]) -> Path:
    return CLIPS_DIR / str(clip["episode_uuid"]) / f"clip_{int(clip['clip_index']):04d}.mp4"


def raw_clip_video_url(clip: dict[str, Any]) -> str | None:
    if clip.get("input_kind") != "anchor":
        return static_url_from_path(clip.get("local_path"), CLIPS_DIR, "clips")
    return static_url_from_path(str(raw_clip_path(clip)), CLIPS_DIR, "clips")


def clip_input_video_url(clip: dict[str, Any]) -> str | None:
    return static_url_from_clip_input_path(clip.get("local_path")) or clip.get("public_url")


def clip_display_video_url(clip: dict[str, Any], input_video_url: str | None, raw_video_url: str | None) -> str | None:
    if clip.get("input_kind") == "anchor" and clip_anchor_stage(clip) == ANCHOR_STAGE_OFFICIAL:
        return raw_video_url or input_video_url
    return input_video_url or raw_video_url


def require_episode_mutation_lock(uuid: str, lock_token: str | None) -> None:
    episode_lock = active_lock("episode", uuid)
    if lock_token:
        require_lock("episode", uuid, lock_token)
        episode = db.one("SELECT final_status FROM episodes WHERE uuid=?", (uuid,))
        if episode and episode.get("final_status") == "stitching":
            raise LockError(409, "episode is currently stitching", episode_lock)
        return
    raise LockError(423, "episode mutation requires an active episode lock", episode_lock)


def acquire_stitch_locks(uuid: str, clips: list[dict[str, Any]]) -> list[str]:
    now = db.now()
    expires_at = now + STITCH_LOCK_TTL_SEC
    resources = [("episode", uuid), *[("clip", str(clip["id"])) for clip in clips]]
    tokens: list[str] = []
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for resource_type, resource_id in resources:
            conn.execute("DELETE FROM resource_locks WHERE resource_type=? AND resource_id=?", (resource_type, resource_id))
            token = secrets.token_urlsafe(24)
            tokens.append(token)
            conn.execute(
                """
                INSERT INTO resource_locks(resource_type, resource_id, owner_id, owner_name, token, expires_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (resource_type, resource_id, STITCH_LOCK_OWNER_ID, STITCH_LOCK_OWNER_NAME, token, expires_at, now, now),
            )
    return tokens


def release_stitch_locks(tokens: list[str]) -> None:
    if not tokens:
        return
    with db.connect() as conn:
        conn.execute(
            "DELETE FROM resource_locks WHERE owner_id=? AND token IN (%s)" % ",".join("?" for _ in tokens),
            [STITCH_LOCK_OWNER_ID, *tokens],
        )


def preprocess(
    uuids: list[str] | None = None,
    fetch_remote: bool = True,
    lock_tokens: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    settings = load_settings()
    lock_tokens = {key.lower(): value for key, value in (lock_tokens or {}).items()}
    if uuids is None:
        episodes = db.rows("SELECT uuid FROM episodes")
        uuids = [row["uuid"] for row in episodes]
    if not uuids:
        return []
    max_workers = min(3, max(1, len(uuids)))
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(preprocess_one, uuid.lower(), settings, fetch_remote, lock_tokens.get(uuid.lower()))
            for uuid in uuids
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def episode_preprocess_integrity(uuid: str) -> dict[str, Any]:
    episode = db.one("SELECT * FROM episodes WHERE uuid=?", (uuid,))
    if not episode:
        return {"complete": False, "reason": "episode is not submitted"}
    head_value = episode.get("head_video_path")
    if episode.get("status") != "preprocessed":
        return {"complete": False, "reason": f"episode status is {episode.get('status') or 'missing'}"}
    if not head_value:
        return {"complete": False, "reason": "head video path is missing"}
    head_path = Path(head_value)
    if not head_path.exists():
        return {"complete": False, "reason": "head video file is missing"}
    try:
        duration = video_duration(head_path)
    except Exception as exc:
        return {"complete": False, "reason": f"head video is invalid: {exc}"}
    return {"complete": True, "reason": "head video already ready", "duration_sec": duration, "clip_count": 0}


def preprocess_one(uuid: str, settings: dict[str, Any], fetch_remote: bool, lock_token: str | None = None) -> dict[str, Any]:
    now = db.now()
    local_dir = EPISODES_DIR / uuid
    require_episode_mutation_lock(uuid, lock_token)
    integrity = episode_preprocess_integrity(uuid)
    if integrity["complete"]:
        return {
            "uuid": uuid,
            "status": "skipped",
            "reason": integrity["reason"],
            "duration_sec": integrity.get("duration_sec"),
            "clip_count": 0,
            "clips": [],
        }
    with db.connect() as conn:
        conn.execute("UPDATE episodes SET status='preprocessing', error=NULL, updated_at=? WHERE uuid=?", (now, uuid))
    try:
        episode_dir = local_dir
        if fetch_remote:
            episode_dir = fetch_episode(settings["dm3_host"], settings["dm3_nedf_root"], uuid, local_dir)
        preprocessed_dir = episode_dir / "preprocessed"
        head_path = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        existing_head = db.one("SELECT head_video_path FROM episodes WHERE uuid=?", (uuid,))
        existing_head_path = Path(existing_head["head_video_path"]) if existing_head and existing_head.get("head_video_path") else head_path
        if not (preprocessed_dir / "metadata.json").exists() and existing_head_path.exists():
            duration = video_duration(existing_head_path)
            clear_episode_clip_state(uuid)
            with db.connect() as conn:
                conn.execute(
                    """
                    UPDATE episodes SET status='preprocessed', head_video_path=?, final_status='missing',
                    continuity_state='select_anchor', anchor_clip_id=NULL,
                    error=NULL, updated_at=? WHERE uuid=?
                    """,
                    (str(existing_head_path.resolve()), db.now(), uuid),
                )
            return {
                "uuid": uuid,
                "status": "preprocessed",
                "reason": integrity["reason"],
                "head": {"output_path": str(existing_head_path.resolve()), "duration_sec": duration, "reused": True},
                "clips": [],
            }
        if not (preprocessed_dir / "metadata.json").exists():
            raise RuntimeError(f"Missing preprocessed metadata for {uuid}")
        meta = extract_head_video(preprocessed_dir, head_path)
        duration = video_duration(head_path)
        clear_episode_clip_state(uuid)
        with db.connect() as conn:
            conn.execute(
                """
                UPDATE episodes SET status='preprocessed', head_video_path=?, final_status='missing',
                continuity_state='select_anchor', anchor_clip_id=NULL,
                local_path=?, error=NULL, updated_at=? WHERE uuid=?
                """,
                (str(head_path.resolve()), str(episode_dir.resolve()), db.now(), uuid),
            )
        return {"uuid": uuid, "status": "preprocessed", "reason": integrity["reason"], "head": meta, "clips": []}
    except Exception as exc:
        with db.connect() as conn:
            conn.execute("UPDATE episodes SET status='failed', error=?, updated_at=? WHERE uuid=?", (str(exc), db.now(), uuid))
        return {"uuid": uuid, "error": str(exc)}


def clear_episode_clip_state(uuid: str) -> None:
    with db.connect() as conn:
        conn.execute(
            "DELETE FROM resource_locks WHERE resource_type='clip' AND resource_id IN (SELECT CAST(id AS TEXT) FROM clips WHERE episode_uuid=?)",
            (uuid,),
        )
        conn.execute("DELETE FROM reviews WHERE clip_id IN (SELECT id FROM clips WHERE episode_uuid=?)", (uuid,))
        conn.execute("DELETE FROM generation_jobs WHERE clip_id IN (SELECT id FROM clips WHERE episode_uuid=?)", (uuid,))
        conn.execute("DELETE FROM clips WHERE episode_uuid=?", (uuid,))
        conn.execute(
            """
            UPDATE episodes
            SET anchor_clip_id=NULL, continuity_state='select_anchor',
                final_status='missing', preview_video_path=NULL, preview_status='missing',
                preview_error=NULL, updated_at=?
            WHERE uuid=?
            """,
            (db.now(), uuid),
        )
    for directory in [CLIPS_DIR / uuid, GENERATED_DIR / uuid, ACCEPTED_DIR / uuid]:
        if directory.exists():
            shutil.rmtree(directory)


def delete_clip_rows(uuid: str, clips: list[dict[str, Any]]) -> int:
    if not clips:
        return 0
    ids = [int(clip["id"]) for clip in clips]
    placeholders = ",".join("?" for _ in ids)
    with db.connect() as conn:
        conn.execute(
            f"DELETE FROM resource_locks WHERE resource_type='clip' AND resource_id IN ({placeholders})",
            [str(item) for item in ids],
        )
        conn.execute(f"DELETE FROM reviews WHERE clip_id IN ({placeholders})", ids)
        conn.execute(f"DELETE FROM generation_jobs WHERE clip_id IN ({placeholders})", ids)
        conn.execute(f"DELETE FROM clips WHERE id IN ({placeholders})", ids)
    for clip in clips:
        Path(clip["local_path"]).unlink(missing_ok=True)
        accepted_path = ACCEPTED_DIR / uuid / f"clip_{int(clip['clip_index']):04d}.mp4"
        accepted_path.unlink(missing_ok=True)
        accepted_chronological_path = chronological_accepted_output_path(clip)
        accepted_chronological_path.unlink(missing_ok=True)
        generated_dir = GENERATED_DIR / uuid
        if generated_dir.exists():
            for path in generated_dir.glob(f"clip_{int(clip['clip_index']):04d}_job_*"):
                path.unlink(missing_ok=True)
    return len(clips)


def archive_anchor_candidates(uuid: str, clips: list[dict[str, Any]], reason: str) -> Path | None:
    if not clips:
        return None
    archive_dir = ARCHIVED_ANCHORS_DIR / uuid / f"{int(time.time() * 1000)}_{reason}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []
    for clip in clips:
        clip_index = int(clip["clip_index"])
        item = dict(clip)
        input_path = Path(clip["local_path"])
        if input_path.exists():
            archived_input = archive_dir / f"clip_{clip_index:04d}_input.mp4"
            shutil.copy2(input_path, archived_input)
            item["archived_input_path"] = str(archived_input.resolve())
        jobs = db.rows("SELECT * FROM generation_jobs WHERE clip_id=? ORDER BY created_at", (clip["id"],))
        archived_jobs = []
        for job in jobs:
            archived_job = dict(job)
            for field, suffix in [("payload_path", "payload.json"), ("output_path", "output.mp4")]:
                value = job.get(field)
                if not value:
                    continue
                source = Path(value)
                if not source.exists():
                    continue
                dst = archive_dir / f"clip_{clip_index:04d}_job_{int(job['id'])}_{suffix}"
                shutil.copy2(source, dst)
                archived_job[f"archived_{field}"] = str(dst.resolve())
            archived_jobs.append(archived_job)
        item["jobs"] = archived_jobs
        metadata.append(item)
    (archive_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return archive_dir


def delete_anchor_candidates_except(uuid: str, keep_clip_id: int) -> int:
    clips = db.rows(
        """
        SELECT * FROM clips
        WHERE episode_uuid=? AND input_kind='anchor' AND id<>?
        ORDER BY clip_index
        """,
        (uuid, keep_clip_id),
    )
    archive_anchor_candidates(uuid, clips, "official_anchor_selected")
    return delete_clip_rows(uuid, clips)


def latest_anchor_candidate_archive(uuid: str, reason: str = "official_anchor_selected") -> Path | None:
    archive_root = ARCHIVED_ANCHORS_DIR / uuid
    if not archive_root.exists():
        return None
    candidates = [
        path
        for path in archive_root.iterdir()
        if path.is_dir() and path.name.endswith(f"_{reason}") and (path / "metadata.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def restore_archived_anchor_candidates(uuid: str) -> int:
    archive_dir = latest_anchor_candidate_archive(uuid)
    if not archive_dir:
        return 0
    metadata_path = archive_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(metadata, list):
        return 0

    restored = 0
    clip_dir = CLIPS_DIR / uuid
    generated_dir = GENERATED_DIR / uuid
    clip_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    now = db.now()
    for raw_item in metadata:
        if not isinstance(raw_item, dict):
            continue
        clip_id = int(raw_item.get("id") or 0)
        clip_index = int(raw_item.get("clip_index") or 0)
        if clip_id <= 0:
            continue
        if db.one("SELECT id FROM clips WHERE id=?", (clip_id,)):
            continue
        input_path = Path(str(raw_item.get("archived_input_path") or ""))
        restored_input = clip_dir / f"clip_{clip_index:04d}.mp4"
        if input_path.exists():
            shutil.copy2(input_path, restored_input)
        elif raw_item.get("local_path") and Path(str(raw_item["local_path"])).exists():
            shutil.copy2(Path(str(raw_item["local_path"])), restored_input)
        else:
            continue
        public_url = public_url_for("clips", Path(uuid) / restored_input.name)
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO clips(
                    id, episode_uuid, clip_index, start_sec, duration_sec,
                    source_start_sec, source_duration_sec, overlap_sec, timeline_duration_sec,
                    timeline_start_sec, timeline_end_sec, input_timeline_start_sec, input_timeline_end_sec,
                    direction, input_kind, anchor_stage,
                    local_path, public_url, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clip_id,
                    uuid,
                    clip_index,
                    float(raw_item.get("start_sec") or 0),
                    float(raw_item.get("duration_sec") or ANCHOR_CLIP_DURATION_SEC),
                    float(raw_item.get("source_start_sec") if raw_item.get("source_start_sec") is not None else raw_item.get("start_sec") or 0),
                    float(raw_item.get("source_duration_sec") if raw_item.get("source_duration_sec") is not None else raw_item.get("duration_sec") or ANCHOR_CLIP_DURATION_SEC),
                    float(raw_item.get("overlap_sec") or 0),
                    float(raw_item.get("timeline_duration_sec") if raw_item.get("timeline_duration_sec") is not None else raw_item.get("duration_sec") or ANCHOR_CLIP_DURATION_SEC),
                    float(raw_item.get("timeline_start_sec") if raw_item.get("timeline_start_sec") is not None else raw_item.get("start_sec") or 0),
                    float(raw_item.get("timeline_end_sec") if raw_item.get("timeline_end_sec") is not None else (float(raw_item.get("start_sec") or 0) + float(raw_item.get("duration_sec") or ANCHOR_CLIP_DURATION_SEC))),
                    float(raw_item.get("input_timeline_start_sec") if raw_item.get("input_timeline_start_sec") is not None else raw_item.get("start_sec") or 0),
                    float(raw_item.get("input_timeline_end_sec") if raw_item.get("input_timeline_end_sec") is not None else (float(raw_item.get("start_sec") or 0) + float(raw_item.get("duration_sec") or ANCHOR_CLIP_DURATION_SEC))),
                    str(raw_item.get("direction") or "anchor"),
                    str(raw_item.get("input_kind") or "anchor"),
                    str(raw_item.get("anchor_stage") or ANCHOR_STAGE_REPLACE_ARM),
                    str(restored_input.resolve()),
                    public_url,
                    str(raw_item.get("status") or "pending"),
                    float(raw_item.get("created_at") or now),
                    now,
                ),
            )
        restored += 1
        for raw_job in raw_item.get("jobs") or []:
            if isinstance(raw_job, dict):
                restore_archived_generation_job(uuid, clip_id, clip_index, raw_job)
    return restored


def restore_archived_generation_job(uuid: str, clip_id: int, clip_index: int, raw_job: dict[str, Any]) -> None:
    job_id = int(raw_job.get("id") or 0)
    if job_id <= 0 or db.one("SELECT id FROM generation_jobs WHERE id=?", (job_id,)):
        return
    output_path = None
    archived_output = Path(str(raw_job.get("archived_output_path") or ""))
    if archived_output.exists():
        suffix = archived_output.suffix or ".mp4"
        output_path = GENERATED_DIR / uuid / f"clip_{clip_index:04d}_job_{job_id}_restored{suffix}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archived_output, output_path)
    now = db.now()
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO generation_jobs(
                id, clip_id, mode, requested_duration_sec, operator_id, operator_name,
                prompt, reference_images_json, task_id, status, output_url, output_path,
                error, started_at, completed_at, estimated_total_sec, retry_count,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                clip_id,
                str(raw_job.get("mode") or "mock"),
                int(raw_job.get("requested_duration_sec") or 0),
                raw_job.get("operator_id"),
                raw_job.get("operator_name"),
                raw_job.get("prompt"),
                raw_job.get("reference_images_json"),
                raw_job.get("task_id"),
                str(raw_job.get("status") or "failed"),
                raw_job.get("output_url"),
                str(output_path.resolve()) if output_path else raw_job.get("output_path"),
                raw_job.get("error"),
                raw_job.get("started_at"),
                raw_job.get("completed_at"),
                raw_job.get("estimated_total_sec"),
                int(raw_job.get("retry_count") or 0),
                float(raw_job.get("created_at") or now),
                float(raw_job.get("updated_at") or now),
            ),
        )


def delete_dependent_continuity_clips(clip: dict[str, Any]) -> int:
    uuid = clip["episode_uuid"]
    input_kind = clip.get("input_kind") or ""
    if input_kind == "anchor":
        episode = db.one("SELECT anchor_clip_id FROM episodes WHERE uuid=?", (uuid,))
        is_official_anchor = bool(episode and int(episode.get("anchor_clip_id") or 0) == int(clip["id"]))
        clips = (
            db.rows(
                "SELECT * FROM clips WHERE episode_uuid=? AND input_kind='rolling' ORDER BY clip_index",
                (uuid,),
            )
            if is_official_anchor
            else []
        )
        deleted = delete_clip_rows(uuid, clips)
        if is_official_anchor:
            with db.connect() as conn:
                conn.execute(
                    """
                    UPDATE episodes
                    SET anchor_clip_id=NULL, continuity_state='anchor_candidates',
                        final_status='stale', updated_at=?
                    WHERE uuid=?
                    """,
                    (db.now(), uuid),
                )
            restore_archived_anchor_candidates(uuid)
        return deleted
    if input_kind != "rolling":
        return 0
    direction = str(clip.get("direction") or "forward")
    if direction == "backward":
        clips = db.rows(
            """
            SELECT * FROM clips
            WHERE episode_uuid=? AND input_kind='rolling' AND direction='backward'
              AND timeline_end_sec<=?
              AND id<>?
            ORDER BY timeline_start_sec
            """,
            (uuid, float(clip.get("timeline_start_sec") or 0) + TIME_EPSILON, int(clip["id"])),
        )
    else:
        clips = db.rows(
            """
            SELECT * FROM clips
            WHERE episode_uuid=? AND input_kind='rolling' AND direction='forward'
              AND timeline_start_sec>=?
              AND id<>?
            ORDER BY timeline_start_sec
            """,
            (uuid, float(clip.get("timeline_end_sec") or 0) - TIME_EPSILON, int(clip["id"])),
        )
    return delete_clip_rows(uuid, clips)


def create_clips(uuid: str, head_path: Path, duration: float) -> list[dict[str, Any]]:
    plan = clip_plan(duration)
    clear_episode_clip_state(uuid)
    clip_dir = CLIPS_DIR / uuid
    clip_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for index, (start, clip_duration) in enumerate(plan):
        path = clip_dir / f"clip_{index:04d}.mp4"
        cut_clip(head_path, path, start, clip_duration)
        public_url = public_url_for("clips", Path(uuid) / path.name)
        now = db.now()
        with db.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO clips(
                    episode_uuid, clip_index, start_sec, duration_sec,
                    source_start_sec, source_duration_sec, overlap_sec, timeline_duration_sec, input_kind,
                    local_path, public_url, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, 'split', ?, ?, 'pending', ?, ?)
                """,
                (uuid, index, start, clip_duration, start, clip_duration, clip_duration, str(path.resolve()), public_url, now, now),
            )
            clip_id = cur.lastrowid
        created.append({"id": clip_id, "episode_uuid": uuid, "clip_index": index, "duration_sec": clip_duration, "path": str(path)})
    return created


def import_head_video(uuid: str, source_path: str, lock_token: str | None = None) -> dict[str, Any]:
    uuid = uuid.lower()
    require_episode_mutation_lock(uuid, lock_token)
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"head video not found: {source}")
    now = db.now()
    head_path = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
    if source.resolve() == head_path.resolve():
        pass
    else:
        transcode_760x570(source, head_path)
    duration = video_duration(head_path)
    clear_episode_clip_state(uuid)
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO episodes(uuid, remote_path, local_path, status, head_video_path, final_status, created_at, updated_at)
            VALUES (?, ?, ?, 'preprocessed', ?, 'missing', ?, ?)
            ON CONFLICT(uuid) DO UPDATE SET
                status='preprocessed',
                head_video_path=excluded.head_video_path,
                final_status='missing',
                continuity_state='select_anchor',
                anchor_clip_id=NULL,
                error=NULL,
                updated_at=excluded.updated_at
            """,
            (
                uuid,
                f"imported:{source}",
                str((EPISODES_DIR / uuid).resolve()),
                str(head_path.resolve()),
                now,
                now,
            ),
        )
    return {
        "uuid": uuid,
        "head_video_path": str(head_path.resolve()),
        "duration_sec": duration,
        "clips": [],
    }


def queue_rolling_generation(
    mode: str | None = None,
    dry_run: bool = False,
    operator_id: str | None = None,
    operator_name: str | None = None,
    prompt: str | None = None,
    reference_images: list[str] | None = None,
) -> list[dict[str, Any]]:
    prepared, skipped = prepare_rolling_generation_clips()
    generated = queue_generation_for_selected_clips(
        mode=mode,
        clips=prepared,
        dry_run=dry_run,
        force=False,
        operator_id=operator_id,
        operator_name=operator_name,
        prompt=prompt,
        reference_images=reference_images,
    )
    episode_by_clip = {int(clip["id"]): clip["episode_uuid"] for clip in prepared}
    for item in generated:
        if "clip_id" in item:
            item["episode_uuid"] = episode_by_clip.get(int(item["clip_id"]))
    return generated + skipped


def prepare_rolling_generation_clips() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes = db.rows("SELECT * FROM episodes WHERE status='preprocessed' AND head_video_path IS NOT NULL ORDER BY created_at")
    prepared: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for episode in episodes:
        results = prepare_rolling_clips_for_episode(episode)
        episode_clips = [item["clip"] for item in results if item.get("clip")]
        if episode_clips:
            prepared.extend(episode_clips)
        else:
            skipped.extend(results or [rolling_skip(episode["uuid"], "waiting for anchor candidates")])
    return prepared, skipped


def prepare_rolling_clips_for_episode(episode: dict[str, Any]) -> list[dict[str, Any]]:
    uuid = episode["uuid"]
    head_value = episode.get("head_video_path")
    if not head_value:
        return [rolling_skip(uuid, "head video path missing")]
    head_path = Path(head_value)
    if not head_path.exists():
        return [rolling_skip(uuid, "head video file missing")]
    try:
        duration = continuity_total_duration(head_path)
    except Exception as exc:
        return [rolling_skip(uuid, str(exc))]
    if duration < MIN_SEEDANCE_INPUT_SEC:
        return [rolling_skip(uuid, f"episode is too short for seedance continuity: {duration:.3f}s")]
    clips = db.rows("SELECT * FROM clips WHERE episode_uuid=? ORDER BY timeline_start_sec, clip_index", (uuid,))
    legacy = [clip for clip in clips if clip.get("input_kind") not in CONTINUITY_INPUT_KINDS]
    if legacy:
        return [rolling_skip(uuid, "legacy split clips exist; continuity generation skipped")]
    if not clips:
        update_continuity_state(uuid)
        return [rolling_skip(uuid, "waiting for anchor candidates")]

    results: list[dict[str, Any]] = []
    for clip in clips:
        if clip["status"] not in GENERATION_CANDIDATE_STATUSES:
            continue
        if clip.get("input_kind") not in CONTINUITY_INPUT_KINDS:
            continue
        try:
            ensure_continuity_clip_input(clip)
        except Exception as exc:
            results.append(rolling_skip(uuid, f"cannot rebuild input for clip {clip['id']}: {exc}"))
            continue
        results.append({"episode_uuid": uuid, "clip": db.one("SELECT * FROM clips WHERE id=?", (clip["id"],)) or clip})
    if not results:
        active = next((clip for clip in clips if clip["status"] in {"generated", "flagged", "generating", "preparing"}), None)
        reason = f"waiting for clip {active['id']} status {active['status']}" if active else "no pending continuity clips"
        results.append(rolling_skip(uuid, reason))
    update_continuity_state(uuid)
    return results


def create_anchor_candidates(uuid: str, start_secs: list[float], lock_token: str | None = None) -> dict[str, Any]:
    uuid = uuid.lower()
    require_episode_mutation_lock(uuid, lock_token)
    episode = db.one("SELECT * FROM episodes WHERE uuid=?", (uuid,))
    if not episode:
        raise ValueError("episode not found")
    if episode.get("status") != "preprocessed" or not episode.get("head_video_path"):
        raise ValueError("episode head video is not ready")
    if episode.get("anchor_clip_id"):
        raise ValueError("official anchor already selected")
    head_path = Path(episode["head_video_path"])
    if not head_path.exists():
        raise ValueError("episode head video file missing")
    total = continuity_total_duration(head_path)
    if total < ANCHOR_CLIP_DURATION_SEC:
        raise ValueError(f"episode is too short for a {ANCHOR_CLIP_DURATION_SEC}s anchor")
    settings = load_settings()
    starts, skipped = normalize_anchor_start_candidates(
        start_secs,
        total,
        continuity_overlap(settings),
        continuity_prefer_input(settings),
    )
    existing_keys = {
        int(round(float(row.get("timeline_start_sec") or row.get("start_sec") or 0.0) * 1000))
        for row in db.rows("SELECT timeline_start_sec, start_sec FROM clips WHERE episode_uuid=? AND input_kind='anchor'", (uuid,))
    }
    created = []
    for start in starts:
        if int(round(start * 1000)) in existing_keys:
            skipped.append({"start_sec": start, "reason": "候选已存在"})
            continue
        plan_item = {
            "start_sec": start,
            "duration_sec": float(ANCHOR_CLIP_DURATION_SEC),
            "source_start_sec": start,
            "source_duration_sec": float(ANCHOR_CLIP_DURATION_SEC),
            "overlap_sec": 0.0,
            "timeline_duration_sec": float(ANCHOR_CLIP_DURATION_SEC),
            "timeline_start_sec": start,
            "timeline_end_sec": start + ANCHOR_CLIP_DURATION_SEC,
            "input_timeline_start_sec": start,
            "input_timeline_end_sec": start + ANCHOR_CLIP_DURATION_SEC,
            "input_kind": "anchor",
            "direction": "anchor",
            "anchor_stage": ANCHOR_STAGE_REPLACE_ARM,
        }
        clip = insert_continuity_clip(uuid, plan_item)
        try:
            build_continuity_clip_input(clip)
        except Exception:
            with db.connect() as conn:
                conn.execute("UPDATE clips SET status='generated_failed', updated_at=? WHERE id=?", (db.now(), clip["id"]))
            raise
        with db.connect() as conn:
            conn.execute("UPDATE clips SET status='pending', updated_at=? WHERE id=?", (db.now(), clip["id"]))
        created.append(db.one("SELECT * FROM clips WHERE id=?", (clip["id"],)) or clip)
        existing_keys.add(int(round(start * 1000)))
    if existing_keys:
        with db.connect() as conn:
            conn.execute(
                """
                UPDATE episodes
                SET continuity_state='anchor_candidates', final_status='stale', updated_at=?
                WHERE uuid=?
                """,
                (db.now(), uuid),
            )
        state = "anchor_candidates"
    else:
        state = update_continuity_state(uuid)["state"]
    return {"uuid": uuid, "created": created, "skipped": skipped, "continuity_state": state}


def normalize_anchor_starts(
    start_secs: list[float],
    total_duration: float,
    overlap: float | None = None,
    prefer_input: float | None = None,
) -> list[float]:
    if not start_secs:
        raise ValueError("at least one anchor start is required")
    max_start = max(0.0, total_duration - ANCHOR_CLIP_DURATION_SEC)
    overlap = continuity_overlap() if overlap is None else float(overlap)
    result: list[float] = []
    seen: set[int] = set()
    for value in start_secs:
        start = min(max(0.0, float(int(float(value)))), max_start)
        key = int(round(start * 1000))
        if key in seen:
            continue
        validate_anchor_start(start, total_duration, overlap, prefer_input)
        seen.add(key)
        result.append(start)
    if not result:
        raise ValueError("no valid anchor starts")
    return result


def normalize_anchor_start_candidates(
    start_secs: list[float],
    total_duration: float,
    overlap: float | None = None,
    prefer_input: float | None = None,
) -> tuple[list[float], list[dict[str, Any]]]:
    overlap = continuity_overlap() if overlap is None else float(overlap)
    prefer_input = continuity_prefer_input() if prefer_input is None else float(prefer_input)
    max_start = max(0.0, total_duration - ANCHOR_CLIP_DURATION_SEC)
    result: list[float] = []
    skipped: list[dict[str, Any]] = []
    seen: set[int] = set()
    for value in start_secs:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            skipped.append({"start_sec": value, "reason": "不是数字"})
            continue
        if not math.isfinite(numeric):
            skipped.append({"start_sec": value, "reason": "不是有限数字"})
            continue
        start = min(max(0.0, float(int(numeric))), max_start)
        key = int(round(start * 1000))
        if key in seen:
            skipped.append({"start_sec": start, "reason": "重复起点"})
            continue
        try:
            validate_anchor_start(start, total_duration, overlap, prefer_input)
        except ValueError as exc:
            skipped.append({"start_sec": start, "reason": f"起点不合法：{exc}"})
            continue
        seen.add(key)
        result.append(start)
    if not start_secs:
        skipped.append({"start_sec": None, "reason": "至少需要一个起点"})
    return result, skipped


def math_floor_millis(value: float) -> float:
    return int(value * 1000) / 1000.0


def continuity_total_duration(head_path: Path) -> float:
    total = int(video_duration(head_path) + TIME_EPSILON)
    if total <= 0:
        return 0.0
    return float(total)


def continuity_overlap(settings: dict[str, Any] | None = None) -> float:
    settings = settings or load_settings()
    try:
        overlap = int(round(float(settings.get("continuity_overlap_sec", 1))))
    except (TypeError, ValueError):
        overlap = 1
    return float(max(0, min(overlap, MAX_SEEDANCE_INPUT_SEC - MIN_SEEDANCE_INPUT_SEC)))


def continuity_prefer_input(settings: dict[str, Any] | None = None) -> float:
    settings = settings or load_settings()
    try:
        prefer = int(round(float(settings.get("continuity_prefer_input_sec", MAX_SEEDANCE_INPUT_SEC))))
    except (TypeError, ValueError):
        prefer = MAX_SEEDANCE_INPUT_SEC
    return float(max(MIN_SEEDANCE_INPUT_SEC, min(prefer, MAX_SEEDANCE_INPUT_SEC)))


def choose_timeline_duration(remaining: float, overlap: float, prefer_input: float | None = None) -> float:
    remaining = float(remaining)
    if remaining <= TIME_EPSILON:
        return 0.0
    prefer_input = continuity_prefer_input() if prefer_input is None else float(prefer_input)
    prefer_input = max(MIN_SEEDANCE_INPUT_SEC, min(prefer_input, MAX_SEEDANCE_INPUT_SEC))
    max_timeline = MAX_SEEDANCE_INPUT_SEC - overlap
    min_timeline = max(0.001, MIN_SEEDANCE_INPUT_SEC - overlap)
    if max_timeline <= 0:
        raise ValueError("overlap leaves no room for source video")
    if remaining + overlap <= prefer_input + TIME_EPSILON:
        if remaining + overlap + TIME_EPSILON < MIN_SEEDANCE_INPUT_SEC:
            raise ValueError(f"remaining timeline {remaining:.3f}s cannot make a legal Seedance input")
        return remaining
    target_timeline = max(min_timeline, min(max_timeline, prefer_input - overlap))
    duration = min(target_timeline, remaining)
    tail = remaining - duration
    if 0 < tail < min_timeline:
        borrow = min_timeline - tail
        if duration - borrow >= min_timeline - TIME_EPSILON:
            duration -= borrow
        elif remaining <= max_timeline + TIME_EPSILON:
            duration = remaining
        else:
            raise ValueError(f"cannot split remaining timeline {remaining:.3f}s around preferred input {prefer_input:.3f}s")
    if duration + overlap + TIME_EPSILON < MIN_SEEDANCE_INPUT_SEC or duration + overlap > MAX_SEEDANCE_INPUT_SEC + TIME_EPSILON:
        raise ValueError(f"cannot choose legal continuity duration from remaining {remaining:.3f}s")
    return duration


def validate_anchor_start(start: float, total_duration: float, overlap: float, prefer_input: float | None = None) -> None:
    left_remaining = float(start)
    right_remaining = float(total_duration) - float(start) - ANCHOR_CLIP_DURATION_SEC
    for label, remaining in [("left", left_remaining), ("right", right_remaining)]:
        if remaining <= TIME_EPSILON:
            continue
        try:
            choose_timeline_duration(remaining, overlap, prefer_input)
        except ValueError as exc:
            raise ValueError(
                f"anchor start {start:.0f}s leaves an illegal {label} side of {remaining:.3f}s"
            ) from exc


def clip_timeline_start(clip: dict[str, Any]) -> float:
    value = clip.get("timeline_start_sec")
    if value is None:
        value = clip.get("source_start_sec") if clip.get("source_start_sec") is not None else clip.get("start_sec")
    return float(value or 0.0)


def clip_timeline_end(clip: dict[str, Any]) -> float:
    value = clip.get("timeline_end_sec")
    if value is not None:
        return float(value)
    return clip_timeline_start(clip) + float(clip.get("timeline_duration_sec") or clip.get("duration_sec") or 0.0)


def clip_input_timeline_start(clip: dict[str, Any]) -> float:
    value = clip.get("input_timeline_start_sec")
    if value is None:
        value = clip.get("start_sec")
    return float(value or 0.0)


def continuity_relevant_clips(
    clips: list[dict[str, Any]],
    anchor_clip_id: int | None = None,
) -> list[dict[str, Any]]:
    result = []
    for clip in clips:
        input_kind = clip.get("input_kind")
        if input_kind not in CONTINUITY_INPUT_KINDS:
            continue
        if input_kind == "anchor" and anchor_clip_id and int(clip["id"]) != anchor_clip_id:
            continue
        result.append(clip)
    return sorted(result, key=lambda item: (clip_timeline_start(item), int(item.get("clip_index") or 0)))


def continuity_coverage(
    clips: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted = [clip for clip in continuity_relevant_clips(clips) if clip.get("status") == "accepted"]
    if not accepted:
        return {"coverage_start_sec": None, "coverage_end_sec": None, "accepted_sec": 0.0, "has_gaps": True}
    accepted.sort(key=lambda item: (clip_timeline_start(item), clip_timeline_end(item)))
    coverage_start = clip_timeline_start(accepted[0])
    coverage_end = clip_timeline_end(accepted[0])
    accepted_sec = max(0.0, coverage_end - coverage_start)
    has_gaps = False
    for clip in accepted[1:]:
        start = clip_timeline_start(clip)
        end = clip_timeline_end(clip)
        accepted_sec += max(0.0, end - start)
        if start > coverage_end + TIME_EPSILON:
            has_gaps = True
        coverage_end = max(coverage_end, end)
    return {
        "coverage_start_sec": coverage_start,
        "coverage_end_sec": coverage_end,
        "accepted_sec": accepted_sec,
        "has_gaps": has_gaps,
    }


def continuity_timeline_summary(
    uuid: str,
    clips: list[dict[str, Any]] | None = None,
    episode: dict[str, Any] | None = None,
) -> dict[str, Any]:
    episode = episode or db.one("SELECT * FROM episodes WHERE uuid=?", (uuid,))
    clips = clips if clips is not None else db.rows("SELECT * FROM clips WHERE episode_uuid=? ORDER BY timeline_start_sec, clip_index", (uuid,))
    if not episode:
        return {"state": "prepare_head", "complete": False, "error": "episode not found"}
    total = None
    error = ""
    head_value = episode.get("head_video_path")
    if head_value and Path(head_value).exists():
        try:
            total = continuity_total_duration(Path(head_value))
        except Exception as exc:
            error = str(exc)
    anchor_clip_id = int(episode.get("anchor_clip_id") or 0) or None
    anchor_candidates = [clip for clip in clips if clip.get("input_kind") == "anchor"]
    official_anchor = db.one("SELECT * FROM clips WHERE id=?", (anchor_clip_id,)) if anchor_clip_id else None
    if official_anchor and official_anchor.get("status") != "accepted":
        official_anchor = None
        anchor_clip_id = None
    relevant = continuity_relevant_clips(clips, anchor_clip_id)
    coverage = continuity_coverage(relevant)
    complete = bool(
        total is not None
        and official_anchor
        and not coverage["has_gaps"]
        and coverage["coverage_start_sec"] is not None
        and float(coverage["coverage_start_sec"]) <= TIME_EPSILON
        and float(coverage["coverage_end_sec"]) >= float(total) - TIME_EPSILON
        and all(clip.get("status") == "accepted" for clip in relevant)
    )
    final_status = episode.get("final_status") or "missing"
    if episode.get("status") != "preprocessed" or not head_value:
        state = "prepare_head"
    elif not official_anchor:
        state = "anchor_candidates" if anchor_candidates else "select_anchor"
    elif final_status == "ready" and complete:
        state = "complete"
    elif final_status == "stitching" and complete:
        state = "stitching"
    elif complete:
        state = "ready_to_stitch"
    else:
        state = "bidirectional"
    return {
        "state": state,
        "complete": complete,
        "total_sec": total,
        "anchor_clip_id": anchor_clip_id,
        "anchor_candidate_count": len(anchor_candidates),
        "relevant_clip_count": len(relevant),
        "error": error,
        **coverage,
    }


def update_continuity_state(uuid: str) -> dict[str, Any]:
    summary = continuity_timeline_summary(uuid)
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE episodes
            SET continuity_state=?, anchor_clip_id=?, updated_at=?
            WHERE uuid=?
            """,
            (summary["state"], summary.get("anchor_clip_id"), db.now(), uuid),
        )
    return summary


def trim_continuity_contribution(
    clip: dict[str, Any],
    accepted_path: Path,
    dst: Path,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    accepted_path = ensure_chronological_accepted_path(clip, accepted_path, should_cancel=should_cancel)
    timeline_start = clip_timeline_start(clip)
    timeline_end = clip_timeline_end(clip)
    duration = max(0.0, timeline_end - timeline_start)
    trim_start = max(0.0, timeline_start - clip_input_timeline_start(clip))
    if duration <= TIME_EPSILON:
        raise ValueError(f"clip {clip['id']} has empty timeline contribution")
    if trim_start <= TIME_EPSILON and abs(duration - float(clip.get("duration_sec") or duration)) <= TIME_EPSILON:
        return accepted_path
    trim_video(accepted_path, dst, trim_start, duration, should_cancel=should_cancel)
    return dst


def stitchable_clips(uuid: str, clips: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    clips = clips if clips is not None else db.rows("SELECT * FROM clips WHERE episode_uuid=? ORDER BY clip_index", (uuid,))
    continuity = [clip for clip in clips if clip.get("input_kind") in CONTINUITY_INPUT_KINDS]
    if not continuity:
        return sorted(clips, key=lambda item: int(item.get("clip_index") or 0))
    summary = continuity_timeline_summary(uuid, clips)
    return continuity_relevant_clips(clips, int(summary["anchor_clip_id"]) if summary.get("anchor_clip_id") else None)


def maybe_prepare_next_continuity_clips_after_accept(
    uuid: str,
    accepted_clip_id: int,
    accepted_path: str | None,
) -> list[dict[str, Any]]:
    if not accepted_path:
        return []
    accepted_clip = db.one("SELECT * FROM clips WHERE id=?", (accepted_clip_id,))
    if not accepted_clip or accepted_clip.get("input_kind") not in CONTINUITY_INPUT_KINDS:
        return []
    if accepted_clip.get("input_kind") == "anchor":
        deleted = delete_anchor_candidates_except(uuid, accepted_clip_id)
        with db.connect() as conn:
            conn.execute(
                """
                UPDATE episodes
                SET anchor_clip_id=?, continuity_state='bidirectional', final_status='stale', updated_at=?
                WHERE uuid=?
                """,
                (accepted_clip_id, db.now(), uuid),
            )
        prepared = []
        for direction in ["backward", "forward"]:
            clip = maybe_prepare_direction_clip(uuid, direction)
            if clip:
                prepared.append(clip)
        update_continuity_state(uuid)
        if deleted:
            for item in prepared:
                item["deleted_anchor_candidate_count"] = deleted
        return prepared
    if accepted_clip.get("input_kind") == "rolling":
        direction = str(accepted_clip.get("direction") or "forward")
        clip = maybe_prepare_direction_clip(uuid, direction)
        update_continuity_state(uuid)
        return [clip] if clip else []
    return []


def maybe_prepare_direction_clip(uuid: str, direction: str) -> dict[str, Any] | None:
    if direction not in {"forward", "backward"}:
        return None
    episode = db.one("SELECT * FROM episodes WHERE uuid=?", (uuid,))
    if not episode or not episode.get("head_video_path") or not episode.get("anchor_clip_id"):
        return None
    head_path = Path(episode["head_video_path"])
    if not head_path.exists():
        return None
    total = continuity_total_duration(head_path)
    anchor_id = int(episode["anchor_clip_id"])
    anchor = db.one("SELECT * FROM clips WHERE id=? AND status='accepted'", (anchor_id,))
    if not anchor:
        return None
    existing_open = db.one(
        """
        SELECT * FROM clips
        WHERE episode_uuid=? AND input_kind='rolling' AND direction=?
          AND status IN ('pending','generated_failed','rejected','generating','generated','flagged','preparing')
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (uuid, direction),
    )
    if existing_open:
        return None
    accepted = db.rows(
        """
        SELECT * FROM clips
        WHERE episode_uuid=? AND input_kind IN ('anchor','rolling') AND status='accepted'
        ORDER BY timeline_start_sec, clip_index
        """,
        (uuid,),
    )
    settings = load_settings()
    overlap = continuity_overlap(settings)
    prefer_input = continuity_prefer_input(settings)
    if direction == "forward":
        boundary = max(float(clip.get("timeline_end_sec") or 0) for clip in accepted)
        remaining = total - boundary
        if remaining <= TIME_EPSILON:
            return None
        timeline_duration = choose_timeline_duration(remaining, overlap, prefer_input)
        timeline_start = boundary
        timeline_end = boundary + timeline_duration
        source_start = timeline_start
        input_timeline_start = timeline_start - overlap
        input_timeline_end = timeline_end
    else:
        boundary = min(float(clip.get("timeline_start_sec") or 0) for clip in accepted)
        remaining = boundary
        if remaining <= TIME_EPSILON:
            return None
        timeline_duration = choose_timeline_duration(remaining, overlap, prefer_input)
        timeline_start = boundary - timeline_duration
        timeline_end = boundary
        source_start = timeline_start
        input_timeline_start = timeline_start
        input_timeline_end = timeline_end + overlap
    plan_item = {
        "start_sec": input_timeline_start,
        "duration_sec": timeline_duration + overlap,
        "source_start_sec": source_start,
        "source_duration_sec": timeline_duration,
        "overlap_sec": overlap,
        "timeline_duration_sec": timeline_duration,
        "timeline_start_sec": timeline_start,
        "timeline_end_sec": timeline_end,
        "input_timeline_start_sec": input_timeline_start,
        "input_timeline_end_sec": input_timeline_end,
        "input_kind": "rolling",
        "direction": direction,
    }
    clip = insert_continuity_clip(uuid, plan_item)
    try:
        build_continuity_clip_input(clip)
    except Exception as exc:
        with db.connect() as conn:
            conn.execute("UPDATE clips SET status='generated_failed', updated_at=? WHERE id=?", (db.now(), clip["id"]))
        item = db.one("SELECT * FROM clips WHERE id=?", (clip["id"],)) or clip
        item["error"] = str(exc)
        return item
    with db.connect() as conn:
        conn.execute("UPDATE clips SET status='pending', updated_at=? WHERE id=?", (db.now(), clip["id"]))
    return db.one("SELECT * FROM clips WHERE id=?", (clip["id"],)) or clip


def rolling_skip(uuid: str, reason: str) -> dict[str, Any]:
    return {"episode_uuid": uuid, "status": "skipped", "reason": reason}


def insert_continuity_clip(uuid: str, plan_item: dict[str, Any]) -> dict[str, Any]:
    clip_index = int(plan_item.get("clip_index") or next_clip_index(uuid))
    path = CLIPS_DIR / uuid / f"clip_{clip_index:04d}.mp4"
    public_url = public_url_for("clips", Path(uuid) / path.name)
    now = db.now()
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT * FROM clips WHERE episode_uuid=? AND clip_index=?",
            (uuid, clip_index),
        ).fetchone()
        if existing:
            return dict(existing)
        cur = conn.execute(
            """
            INSERT INTO clips(
                episode_uuid, clip_index, start_sec, duration_sec,
                source_start_sec, source_duration_sec, overlap_sec, timeline_duration_sec,
                timeline_start_sec, timeline_end_sec, input_timeline_start_sec, input_timeline_end_sec,
                direction, input_kind, anchor_stage,
                local_path, public_url, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'preparing', ?, ?)
            """,
            (
                uuid,
                clip_index,
                float(plan_item["start_sec"]),
                float(plan_item["duration_sec"]),
                float(plan_item["source_start_sec"]),
                float(plan_item["source_duration_sec"]),
                float(plan_item["overlap_sec"]),
                float(plan_item["timeline_duration_sec"]),
                float(plan_item["timeline_start_sec"]),
                float(plan_item["timeline_end_sec"]),
                float(plan_item["input_timeline_start_sec"]),
                float(plan_item["input_timeline_end_sec"]),
                str(plan_item.get("direction") or "forward"),
                str(plan_item.get("input_kind") or "rolling"),
                str(plan_item.get("anchor_stage") or ""),
                str(path.resolve()),
                public_url,
                now,
                now,
            ),
        )
        clip_id = cur.lastrowid
        conn.execute(
            "UPDATE episodes SET final_status='stale', updated_at=? WHERE uuid=? AND final_status IN ('ready','stitching')",
            (now, uuid),
        )
    return db.one("SELECT * FROM clips WHERE id=?", (clip_id,)) or {"id": clip_id, "episode_uuid": uuid}


def next_clip_index(uuid: str) -> int:
    row = db.one("SELECT COALESCE(MAX(clip_index), -1) + 1 AS next_index FROM clips WHERE episode_uuid=?", (uuid,))
    return int(row["next_index"] if row else 0)


def ensure_continuity_clip_input(clip: dict[str, Any]) -> None:
    path = Path(clip["local_path"])
    if path.exists():
        return
    build_continuity_clip_input(clip)


def build_continuity_clip_input(clip: dict[str, Any]) -> None:
    episode = db.one("SELECT * FROM episodes WHERE uuid=?", (clip["episode_uuid"],))
    if not episode or not episode.get("head_video_path"):
        raise RuntimeError("episode head video path missing")
    head_path = Path(episode["head_video_path"])
    if not head_path.exists():
        raise RuntimeError("episode head video file missing")
    anchor_path = None
    overlap = float(clip.get("overlap_sec") or 0)
    direction = str(clip.get("direction") or "forward")
    if overlap > 0:
        anchor_path = adjacent_accepted_path(clip)
        if not anchor_path:
            raise RuntimeError("adjacent accepted output is missing")
    if clip.get("input_kind") == "anchor":
        direction = "forward"
    compose_continuity_input(
        head_path,
        Path(clip["local_path"]),
        float(clip.get("source_start_sec") if clip.get("source_start_sec") is not None else clip["start_sec"]),
        float(clip.get("source_duration_sec") if clip.get("source_duration_sec") is not None else clip["duration_sec"]),
        direction,
        anchor_path,
        overlap,
    )


def adjacent_accepted_path(clip: dict[str, Any]) -> Path | None:
    direction = str(clip.get("direction") or "forward")
    if direction == "backward":
        neighbor = db.one(
            """
            SELECT * FROM clips
            WHERE episode_uuid=? AND input_kind IN ('anchor','rolling') AND status='accepted'
              AND timeline_start_sec>=?
            ORDER BY timeline_start_sec ASC
            LIMIT 1
            """,
            (clip["episode_uuid"], float(clip.get("timeline_end_sec") or 0) - TIME_EPSILON),
        )
    else:
        neighbor = db.one(
            """
            SELECT * FROM clips
            WHERE episode_uuid=? AND input_kind IN ('anchor','rolling') AND status='accepted'
              AND timeline_end_sec<=?
            ORDER BY timeline_end_sec DESC
            LIMIT 1
            """,
            (clip["episode_uuid"], float(clip.get("timeline_start_sec") or 0) + TIME_EPSILON),
        )
    return latest_accepted_path(int(neighbor["id"])) if neighbor else None


def latest_accepted_path(clip_id: int) -> Path | None:
    clip = db.one("SELECT * FROM clips WHERE id=?", (clip_id,))
    if not clip:
        return None
    review = db.one(
        "SELECT * FROM reviews WHERE clip_id=? AND decision='accept' ORDER BY reviewed_at DESC LIMIT 1",
        (clip_id,),
    )
    if not review or not review.get("accepted_path"):
        return None
    path = Path(review["accepted_path"])
    if not path.exists():
        return None
    return ensure_chronological_accepted_path(clip, path)


def queue_generation_for_selected_clips(
    mode: str | None,
    clips: list[dict[str, Any]],
    dry_run: bool,
    force: bool,
    operator_id: str | None,
    operator_name: str | None,
    prompt: str | None,
    reference_images: list[str] | None,
) -> list[dict[str, Any]]:
    settings = load_settings()
    mode = mode or DEFAULT_REQUEST_MODE
    if mode not in {"mock", "seedance"}:
        raise ValueError("mode must be mock or seedance")
    operator_id, operator_name = normalize_operator(operator_id, operator_name)
    available = filter_generation_clips(clips, {}, strict=False, operator_id=operator_id)
    if not available:
        return []
    if dry_run or (mode == "mock" and not settings.get("mock_async")):
        client = SeedanceClient(settings)
        concurrency_key = "mock_concurrency" if mode == "mock" or dry_run else "seedance_concurrency"
        max_workers = min(max(1, int(settings.get(concurrency_key, 1))), len(available))
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    run_generation_for_clip,
                    client,
                    clip,
                    mode,
                    settings,
                    dry_run,
                    force,
                    operator_id,
                    operator_name,
                    *generation_values_for_clip(settings, clip, prompt, reference_images),
                )
                for clip in available
            ]
            for future in as_completed(futures):
                results.append(future.result())
        return results

    claimed = []
    for clip in available:
        generation_prompt, generation_refs = generation_values_for_clip(settings, clip, prompt, reference_images)
        item = claim_async_generation_job(clip, mode, settings, force, operator_id, operator_name, generation_prompt, generation_refs)
        if item.get("status") == "queued":
            claimed.append(item)
    for item in claimed:
        if mode == "mock":
            _GENERATION_EXECUTOR.submit(mock_job_worker, int(item["job_id"]))
        else:
            _GENERATION_EXECUTOR.submit(seedance_job_worker, int(item["job_id"]))
    return claimed


def run_generation(
    mode: str | None = None,
    clip_ids: list[int] | None = None,
    dry_run: bool = False,
    lock_tokens: dict[str, str] | None = None,
    force: bool = False,
    operator_id: str | None = None,
    operator_name: str | None = None,
    prompt: str | None = None,
    reference_images: list[str] | None = None,
) -> list[dict[str, Any]]:
    settings = load_settings()
    mode = mode or DEFAULT_REQUEST_MODE
    if mode not in {"mock", "seedance"}:
        raise ValueError("mode must be mock or seedance")
    if clip_ids is not None and len(clip_ids) == 0:
        return []
    if clip_ids is not None:
        clips = db.rows("SELECT * FROM clips WHERE id IN (%s)" % ",".join("?" for _ in clip_ids), clip_ids)
    else:
        clips = db.rows(
            f"SELECT * FROM clips WHERE status IN ({GENERATION_CANDIDATE_STATUS_PLACEHOLDERS}) ORDER BY episode_uuid, clip_index",
            GENERATION_CANDIDATE_STATUSES,
        )
    operator_id, operator_name = normalize_operator(operator_id, operator_name)
    clips = filter_generation_clips(
        clips,
        lock_tokens or {},
        strict=clip_ids is not None,
        operator_id=operator_id,
    )
    if not clips:
        return []
    client = SeedanceClient(settings)
    concurrency_key = "mock_concurrency" if mode == "mock" or dry_run else "seedance_concurrency"
    max_workers = max(1, int(settings.get(concurrency_key, 1)))
    max_workers = min(max_workers, len(clips))
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_generation_for_clip,
                client,
                clip,
                mode,
                settings,
                dry_run,
                force,
                operator_id,
                operator_name,
                *generation_values_for_clip(settings, clip, prompt, reference_images),
            )
            for clip in clips
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def queue_generation(
    mode: str | None = None,
    clip_ids: list[int] | None = None,
    dry_run: bool = False,
    lock_tokens: dict[str, str] | None = None,
    force: bool = False,
    operator_id: str | None = None,
    operator_name: str | None = None,
    prompt: str | None = None,
    reference_images: list[str] | None = None,
) -> list[dict[str, Any]]:
    settings = load_settings()
    mode = mode or DEFAULT_REQUEST_MODE
    if dry_run:
        return run_generation(mode, clip_ids, dry_run, lock_tokens, force, operator_id, operator_name, prompt, reference_images)
    if mode == "mock" and not settings.get("mock_async"):
        return run_generation(mode, clip_ids, dry_run, lock_tokens, force, operator_id, operator_name, prompt, reference_images)
    if mode not in {"mock", "seedance"}:
        raise ValueError("mode must be mock or seedance")
    if clip_ids is not None and len(clip_ids) == 0:
        return []
    if clip_ids is not None:
        clips = db.rows("SELECT * FROM clips WHERE id IN (%s)" % ",".join("?" for _ in clip_ids), clip_ids)
    else:
        clips = db.rows(
            f"SELECT * FROM clips WHERE status IN ({GENERATION_CANDIDATE_STATUS_PLACEHOLDERS}) ORDER BY episode_uuid, clip_index",
            GENERATION_CANDIDATE_STATUSES,
        )
    operator_id, operator_name = normalize_operator(operator_id, operator_name)
    clips = filter_generation_clips(
        clips,
        lock_tokens or {},
        strict=clip_ids is not None,
        operator_id=operator_id,
    )
    if not clips:
        return []
    claimed = []
    for clip in clips:
        generation_prompt, generation_refs = generation_values_for_clip(settings, clip, prompt, reference_images)
        item = claim_async_generation_job(clip, mode, settings, force, operator_id, operator_name, generation_prompt, generation_refs)
        if item.get("status") == "queued":
            claimed.append(item)
    for item in claimed:
        if mode == "mock":
            _GENERATION_EXECUTOR.submit(mock_job_worker, int(item["job_id"]))
        else:
            _GENERATION_EXECUTOR.submit(seedance_job_worker, int(item["job_id"]))
    return claimed


def claim_async_generation_job(
    clip: dict[str, Any],
    mode: str,
    settings: dict[str, Any],
    force: bool = False,
    operator_id: str = "",
    operator_name: str = "",
    prompt: str = "",
    reference_images: list[str] | None = None,
) -> dict[str, Any]:
    requested = requested_seedance_duration(float(clip["duration_sec"]))
    prompt, reference_images = generation_overrides(settings, prompt, reference_images)
    if mode == "mock":
        estimate = max(0.1, float(clip["duration_sec"]) * float(settings.get("mock_seconds_per_video_second") or 0.25))
    else:
        estimate = requested * float(settings.get("seedance_seconds_per_video_second") or 24)
    now = db.now()
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        running = conn.execute(
            "SELECT id FROM generation_jobs WHERE clip_id=? AND status='running' ORDER BY created_at DESC LIMIT 1",
            (clip["id"],),
        ).fetchone()
        if running:
            return {"clip_id": clip["id"], "status": "skipped", "reason": "clip already has a running job"}
        current = conn.execute("SELECT status FROM clips WHERE id=?", (clip["id"],)).fetchone()
        if not current:
            return {"clip_id": clip["id"], "status": "skipped", "reason": "clip not found"}
        can_claim = current["status"] != "generating" if force else current["status"] in GENERATION_CANDIDATE_STATUSES
        if not can_claim:
            return {"clip_id": clip["id"], "status": "skipped", "reason": f"clip status is {current['status']}"}
        cur = conn.execute(
            """
            INSERT INTO generation_jobs(
                clip_id, mode, requested_duration_sec, operator_id, operator_name,
                prompt, reference_images_json, status, retry_count,
                started_at, estimated_total_sec, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'running', 0, ?, ?, ?, ?)
            """,
            (
                clip["id"],
                mode,
                requested,
                operator_id,
                operator_name,
                prompt,
                json_text(reference_images or []),
                now,
                estimate,
                now,
                now,
            ),
        )
        job_id = cur.lastrowid
        conn.execute("UPDATE clips SET status='generating', updated_at=? WHERE id=?", (now, clip["id"]))
        conn.execute(
            "UPDATE episodes SET final_status='stale', updated_at=? WHERE uuid=? AND final_status IN ('ready','stitching')",
            (now, clip["episode_uuid"]),
        )
    return {"job_id": job_id, "clip_id": clip["id"], "status": "queued", "estimated_total_sec": estimate}


def mock_job_worker(job_id: int) -> None:
    job = db.one(
        """
        SELECT j.*, c.episode_uuid, c.clip_index, c.local_path
        FROM generation_jobs j
        JOIN clips c ON c.id = j.clip_id
        WHERE j.id=?
        """,
        (job_id,),
    )
    if not job or job.get("status") != "running":
        return
    out_path = GENERATED_DIR / job["episode_uuid"] / f"clip_{int(job['clip_index']):04d}_job_{job_id}_mock.mp4"
    try:
        remaining = max(0.0, float(job.get("estimated_total_sec") or 0))
        while remaining > 0:
            sleep_for = min(0.5, remaining)
            time.sleep(sleep_for)
            remaining -= sleep_for
            with db.connect() as conn:
                conn.execute("UPDATE generation_jobs SET updated_at=? WHERE id=? AND status='running'", (db.now(), job_id))
        data = SeedanceClient(load_settings()).mock_generate(Path(job["local_path"]), out_path)
        with db.connect() as conn:
            now = db.now()
            conn.execute(
                """
                UPDATE generation_jobs
                SET status='succeeded', task_id=?, output_url=?, output_path=?, error=NULL, completed_at=?, updated_at=?
                WHERE id=?
                """,
                (data["task_id"], data["output_url"], str(out_path.resolve()), now, now, job_id),
            )
            conn.execute("UPDATE clips SET status='generated', updated_at=? WHERE id=? AND status='generating'", (now, job["clip_id"]))
    except Exception as exc:
        fail_generation_job(job_id, int(job["clip_id"]), str(exc))


def seedance_job_worker(job_id: int) -> None:
    if not begin_generation_worker(job_id):
        return
    key_slot: dict[str, Any] | None = None
    job: dict[str, Any] | None = None
    try:
        row = db.one(
            """
            SELECT j.*, c.episode_uuid, c.clip_index, c.duration_sec, c.public_url
            FROM generation_jobs j
            JOIN clips c ON c.id = j.clip_id
            WHERE j.id=?
            """,
            (job_id,),
        )
        if not row or row.get("status") != "running":
            return
        job = dict(row)
        settings = load_settings()
        job_prompt, job_refs = generation_values_from_job(job, settings)
        out_path = GENERATED_DIR / job["episode_uuid"] / f"clip_{int(job['clip_index']):04d}_job_{job_id}_seedance.mp4"
        heartbeat = generation_job_heartbeat(job_id)
        key_slot = acquire_seedance_key_slot(settings)
        client = SeedanceClient({**settings, "seedance_api_key": key_slot["api_key"]})
        job["api_key_id"] = key_slot["id"]
        job["api_key_name"] = key_slot["name"]
        if not job.get("task_id"):
            try:
                task = client.create_task(job_prompt, job["public_url"], float(job["duration_sec"]), job_refs)
            except Exception as exc:
                record_seedance_api_call(job, "failed", error=str(exc))
                raise
            job["task_id"] = task["task_id"]
            record_seedance_api_call(
                job,
                "submitted",
                task_id=job["task_id"],
                usage=task.get("usage"),
                raw_response=task.get("raw_response"),
            )
            with db.connect() as conn:
                conn.execute("UPDATE generation_jobs SET task_id=?, updated_at=? WHERE id=?", (job["task_id"], db.now(), job_id))
        data = client.wait_for_task(
            job["task_id"],
            out_path,
            input_url=job["public_url"],
            on_poll=lambda _task: heartbeat(),
            on_download_progress=lambda _received, _expected: heartbeat(),
        )
        update_seedance_api_call(
            job,
            "succeeded",
            task_id=job["task_id"],
            usage=data.get("usage"),
            raw_response=data.get("raw_response"),
        )
        with db.connect() as conn:
            now = db.now()
            conn.execute(
                """
                UPDATE generation_jobs
                SET status='succeeded', output_url=?, output_path=?, error=NULL, completed_at=?, updated_at=?
                WHERE id=?
                """,
                (data["output_url"], str(out_path.resolve()), now, now, job_id),
            )
            conn.execute("UPDATE clips SET status='generated', updated_at=? WHERE id=? AND status='generating'", (now, job["clip_id"]))
    except Exception as exc:
        if job and job.get("task_id"):
            update_seedance_api_call(job, "failed", task_id=job.get("task_id"), error=str(exc))
        fail_generation_job(job_id, int(job["clip_id"]) if job else 0, str(exc))
    finally:
        release_seedance_key_slot(key_slot)
        end_generation_worker(job_id)


def begin_generation_worker(job_id: int) -> bool:
    with _GENERATION_WORKER_LOCK:
        if job_id in _GENERATION_WORKER_JOB_IDS:
            return False
        _GENERATION_WORKER_JOB_IDS.add(job_id)
        return True


def end_generation_worker(job_id: int) -> None:
    with _GENERATION_WORKER_LOCK:
        _GENERATION_WORKER_JOB_IDS.discard(job_id)


def generation_worker_is_active(job_id: int) -> bool:
    with _GENERATION_WORKER_LOCK:
        return job_id in _GENERATION_WORKER_JOB_IDS


def generation_job_heartbeat(job_id: int, min_interval_sec: float = 5.0) -> Callable[[], None]:
    last = 0.0

    def touch() -> None:
        nonlocal last
        now = db.now()
        if now - last < min_interval_sec:
            return
        last = now
        with db.connect() as conn:
            conn.execute("UPDATE generation_jobs SET updated_at=? WHERE id=? AND status='running'", (now, job_id))

    return touch


def active_seedance_key_pool(settings: dict[str, Any]) -> list[dict[str, Any]]:
    pool = [
        item
        for item in seedance_api_key_pool(settings)
        if item.get("enabled", True) and item.get("api_key") and int(item.get("concurrency") or 0) > 0
    ]
    if not pool:
        raise RuntimeError("seedance_api_key is required for seedance mode")
    return pool


def acquire_seedance_key_slot(settings: dict[str, Any]) -> dict[str, Any]:
    global _GENERATION_ACTIVE
    pool = active_seedance_key_pool(settings)
    with _GENERATION_CONDITION:
        while True:
            available = []
            for item in pool:
                key_id = str(item["id"])
                active = _SEEDANCE_KEY_ACTIVE.get(key_id, 0)
                limit = max(1, int(item.get("concurrency") or 1))
                if active < limit:
                    available.append((active / limit, active, item))
            if available:
                _, _, chosen = min(available, key=lambda value: (value[0], value[1], str(value[2]["id"])))
                key_id = str(chosen["id"])
                _SEEDANCE_KEY_ACTIVE[key_id] = _SEEDANCE_KEY_ACTIVE.get(key_id, 0) + 1
                _GENERATION_ACTIVE += 1
                return dict(chosen)
            _GENERATION_CONDITION.wait(timeout=5)


def release_seedance_key_slot(key_slot: dict[str, Any] | None) -> None:
    global _GENERATION_ACTIVE
    with _GENERATION_CONDITION:
        if not key_slot:
            _GENERATION_CONDITION.notify_all()
            return
        key_id = str(key_slot.get("id") or "")
        if key_id:
            next_active = max(0, _SEEDANCE_KEY_ACTIVE.get(key_id, 0) - 1)
            if next_active:
                _SEEDANCE_KEY_ACTIVE[key_id] = next_active
            else:
                _SEEDANCE_KEY_ACTIVE.pop(key_id, None)
        _GENERATION_ACTIVE = max(0, _GENERATION_ACTIVE - 1)
        _GENERATION_CONDITION.notify_all()


def fail_generation_job(job_id: int, clip_id: int, error: str) -> None:
    with db.connect() as conn:
        now = db.now()
        conn.execute(
            "UPDATE generation_jobs SET status='failed', error=?, completed_at=?, updated_at=? WHERE id=?",
            (error, now, now, job_id),
        )
        conn.execute("UPDATE clips SET status='generated_failed', updated_at=? WHERE id=? AND status='generating'", (now, clip_id))


def recover_interrupted_generation_jobs(stale_after_sec: float = 0.0, include_download_failures: bool = False) -> dict[str, list[int]]:
    now = db.now()
    running_jobs = db.rows(
        """
        SELECT j.*, c.status AS clip_status
        FROM generation_jobs j
        JOIN clips c ON c.id = j.clip_id
        WHERE j.status='running'
          AND (? <= 0 OR j.updated_at <= ?)
        ORDER BY j.created_at
        """,
        (stale_after_sec, now - max(0.0, stale_after_sec)),
    )
    failed_jobs = (
        db.rows(
            """
            SELECT j.*, c.status AS clip_status
            FROM generation_jobs j
            JOIN clips c ON c.id = j.clip_id
            WHERE j.status='failed'
              AND j.mode='seedance'
              AND j.task_id IS NOT NULL
              AND j.retry_count < ?
              AND c.status='generated_failed'
              AND j.created_at = (
                  SELECT MAX(j2.created_at)
                  FROM generation_jobs j2
                  WHERE j2.clip_id=j.clip_id
              )
            ORDER BY j.updated_at
            """,
            (GENERATION_DOWNLOAD_RECOVERY_LIMIT,),
        )
        if include_download_failures
        else []
    )
    resumed: list[int] = []
    failed: list[int] = []
    for job in running_jobs:
        job_id = int(job["id"])
        clip_id = int(job["clip_id"])
        if generation_worker_is_active(job_id):
            continue
        if job.get("mode") == "seedance" and job.get("task_id"):
            _GENERATION_EXECUTOR.submit(seedance_job_worker, job_id)
            resumed.append(job_id)
            continue
        fail_generation_job(job_id, clip_id, "generation job was interrupted before it could be resumed")
        failed.append(job_id)
    for job in failed_jobs:
        job_id = int(job["id"])
        if generation_worker_is_active(job_id) or not is_recoverable_seedance_download_error(job.get("error")):
            continue
        with db.connect() as conn:
            now = db.now()
            conn.execute(
                """
                UPDATE generation_jobs
                SET status='running', error=NULL, completed_at=NULL,
                    retry_count=retry_count + 1, updated_at=?
                WHERE id=? AND status='failed'
                """,
                (now, job_id),
            )
            conn.execute("UPDATE clips SET status='generating', updated_at=? WHERE id=?", (now, int(job["clip_id"])))
        _GENERATION_EXECUTOR.submit(seedance_job_worker, job_id)
        resumed.append(job_id)
    return {"resumed": resumed, "failed": failed}


def is_recoverable_seedance_download_error(error: str | None) -> bool:
    text = (error or "").lower()
    return any(pattern in text for pattern in ["urlopen", "retrieval incomplete", "download", "incomplete", "timed out"])


def start_generation_watchdog() -> None:
    global _GENERATION_WATCHDOG_STARTED
    with _GENERATION_WORKER_LOCK:
        if _GENERATION_WATCHDOG_STARTED:
            return
        _GENERATION_WATCHDOG_STARTED = True
    thread = threading.Thread(target=generation_watchdog_loop, name="seedance-generation-watchdog", daemon=True)
    thread.start()


def generation_watchdog_loop() -> None:
    while True:
        try:
            recover_interrupted_generation_jobs(
                stale_after_sec=GENERATION_STALE_RUNNING_SEC,
                include_download_failures=True,
            )
        except Exception:
            pass
        time.sleep(GENERATION_WATCHDOG_INTERVAL_SEC)


def filter_generation_clips(
    clips: list[dict[str, Any]],
    lock_tokens: dict[str, str],
    strict: bool,
    operator_id: str = "",
) -> list[dict[str, Any]]:
    active_clip_locks = locks_by_resource("clip")
    active_episode_locks = locks_by_resource("episode")
    available = []
    for clip in clips:
        clip_id = str(clip["id"])
        episode_uuid = str(clip["episode_uuid"])
        clip_token = lock_tokens.get(clip_id)
        episode_token = lock_tokens.get(episode_uuid)
        if episode_uuid in active_episode_locks:
            lock = active_episode_locks[episode_uuid]
            if episode_token:
                require_lock("episode", episode_uuid, episode_token)
            elif operator_id and lock.get("owner_id") == operator_id:
                pass
            elif strict:
                require_lock("episode", episode_uuid, None)
            else:
                continue
        if clip_id in active_clip_locks:
            if clip_token:
                require_lock("clip", clip_id, clip_token)
            elif strict:
                require_lock("clip", clip_id, None)
            else:
                continue
        available.append(clip)
    return available


def normalize_operator(operator_id: str | None, operator_name: str | None) -> tuple[str, str]:
    normalized_id = (operator_id or "").strip()
    normalized_name = (operator_name or "").strip()
    if not normalized_name:
        normalized_name = normalized_id
    return normalized_id, normalized_name


def operator_from_lock_token(lock_token: str | None) -> tuple[str, str]:
    if not lock_token:
        return "", ""
    row = db.one("SELECT owner_id, owner_name FROM resource_locks WHERE token=?", (lock_token,))
    if not row:
        return "", ""
    return normalize_operator(row.get("owner_id"), row.get("owner_name"))


def generation_overrides(
    settings: dict[str, Any],
    prompt: str | None,
    reference_images: list[str] | None,
) -> tuple[str, list[str]]:
    generation_prompt = (prompt or "").strip() or str(settings.get("default_prompt") or "")
    if reference_images is None:
        generation_refs = settings.get("reference_images") or []
    else:
        generation_refs = reference_images
    return generation_prompt, [str(item) for item in generation_refs if str(item).strip()]


def preset_generation_values(settings: dict[str, Any], preset_id: str) -> tuple[str, list[str]]:
    presets = settings.get("generation_presets") if isinstance(settings.get("generation_presets"), list) else []
    preset = next((item for item in presets if str(item.get("id") or "") == preset_id), None)
    if not preset:
        raise RuntimeError(f"generation preset is missing: {preset_id}")
    prompt = str(preset.get("prompt") or "").strip()
    refs = [str(item) for item in (preset.get("reference_images") or []) if str(item).strip()]
    if not prompt:
        raise RuntimeError(f"generation preset prompt is empty: {preset_id}")
    return prompt, refs


def clip_anchor_stage(clip: dict[str, Any]) -> str:
    stage = str(clip.get("anchor_stage") or "").strip()
    if stage:
        return stage
    return ANCHOR_STAGE_REPLACE_ARM if clip.get("input_kind") == "anchor" else ""


def locked_generation_preset_id(clip: dict[str, Any]) -> str | None:
    input_kind = str(clip.get("input_kind") or "")
    if input_kind == "anchor":
        stage = clip_anchor_stage(clip)
        if stage == ANCHOR_STAGE_REPLACE_COLLECTOR or stage == ANCHOR_STAGE_OFFICIAL:
            return COLLECTOR_ONLY_PRESET_ID
        return DEFAULT_GENERATION_PRESET_ID
    if input_kind == "rolling":
        return IPHONE2DEPLOY_PRESET_ID
    return None


def generation_values_for_clip(
    settings: dict[str, Any],
    clip: dict[str, Any],
    prompt: str | None,
    reference_images: list[str] | None,
) -> tuple[str, list[str]]:
    preset_id = locked_generation_preset_id(clip)
    if preset_id:
        return preset_generation_values(settings, preset_id)
    return generation_overrides(settings, prompt, reference_images)


def generation_values_from_job(job: dict[str, Any], settings: dict[str, Any]) -> tuple[str, list[str]]:
    prompt = (job.get("prompt") or "").strip() or str(settings.get("default_prompt") or "")
    refs = parse_json_text(job.get("reference_images_json"))
    if not isinstance(refs, list):
        refs = settings.get("reference_images") or []
    return prompt, [str(item) for item in refs if str(item).strip()]


def parse_json_text(value: Any) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def json_text(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str)


def seedance_call_payload(
    job: dict[str, Any],
    status: str,
    task_id: str | None = None,
    usage: Any = None,
    raw_response: Any = None,
    error: str | None = None,
) -> dict[str, Any]:
    now = db.now()
    return {
        "job_id": job.get("id"),
        "clip_id": job.get("clip_id"),
        "operator_id": job.get("operator_id") or "",
        "operator_name": job.get("operator_name") or "",
        "api_key_id": job.get("api_key_id") or "",
        "api_key_name": job.get("api_key_name") or "",
        "call_type": "create_task",
        "status": status,
        "task_id": task_id or job.get("task_id") or "",
        "model": load_settings().get("seedance_model") or "",
        "requested_duration_sec": job.get("requested_duration_sec"),
        "clip_duration_sec": job.get("duration_sec"),
        "usage_json": json_text(usage),
        "raw_response_json": json_text(raw_response),
        "error": error,
        "created_at": now,
        "updated_at": now,
    }


def record_seedance_api_call(
    job: dict[str, Any],
    status: str,
    task_id: str | None = None,
    usage: Any = None,
    raw_response: Any = None,
    error: str | None = None,
) -> None:
    payload = seedance_call_payload(job, status, task_id, usage, raw_response, error)
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO seedance_api_calls(
                job_id, clip_id, operator_id, operator_name, api_key_id, api_key_name, call_type, status,
                task_id, model, requested_duration_sec, clip_duration_sec,
                usage_json, raw_response_json, error, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["job_id"],
                payload["clip_id"],
                payload["operator_id"],
                payload["operator_name"],
                payload["api_key_id"],
                payload["api_key_name"],
                payload["call_type"],
                payload["status"],
                payload["task_id"],
                payload["model"],
                payload["requested_duration_sec"],
                payload["clip_duration_sec"],
                payload["usage_json"],
                payload["raw_response_json"],
                payload["error"],
                payload["created_at"],
                payload["updated_at"],
            ),
        )


def update_seedance_api_call(
    job: dict[str, Any],
    status: str,
    task_id: str | None = None,
    usage: Any = None,
    raw_response: Any = None,
    error: str | None = None,
) -> None:
    now = db.now()
    call_id = None
    with db.connect() as conn:
        row = conn.execute(
            """
            SELECT id FROM seedance_api_calls
            WHERE job_id=? AND call_type='create_task'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (job.get("id"),),
        ).fetchone()
        call_id = row["id"] if row else None
    if call_id is None:
        record_seedance_api_call(job, status, task_id, usage, raw_response, error)
        return
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE seedance_api_calls
            SET status=?, task_id=?, usage_json=COALESCE(?, usage_json),
                raw_response_json=COALESCE(?, raw_response_json), error=?, updated_at=?
            WHERE id=?
            """,
            (status, task_id or job.get("task_id") or "", json_text(usage), json_text(raw_response), error, now, call_id),
        )


def run_generation_for_clip(
    client: SeedanceClient,
    clip: dict[str, Any],
    mode: str,
    settings: dict[str, Any],
    dry_run: bool,
    force: bool = False,
    operator_id: str = "",
    operator_name: str = "",
    prompt: str = "",
    reference_images: list[str] | None = None,
) -> dict[str, Any]:
    requested = requested_seedance_duration(float(clip["duration_sec"]))
    prompt, reference_images = generation_overrides(settings, prompt, reference_images)
    now = db.now()
    previous_clip_status = ""
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute("SELECT status FROM clips WHERE id=?", (clip["id"],)).fetchone()
        if not current:
            return {"clip_id": clip["id"], "status": "skipped", "reason": "clip not found"}
        previous_clip_status = current["status"]
        can_claim = current["status"] != "generating" if force else current["status"] in GENERATION_CANDIDATE_STATUSES
        if not can_claim:
            return {
                "clip_id": clip["id"],
                "status": "skipped",
                "reason": f"clip status is {current['status']}",
            }
        estimated_total_sec = (
            requested * float(settings.get("seedance_seconds_per_video_second") or 24)
            if mode == "seedance" and not dry_run
            else max(1.0, float(clip["duration_sec"]))
        )
        cur = conn.execute(
            """
            INSERT INTO generation_jobs(
                clip_id, mode, requested_duration_sec, operator_id, operator_name,
                prompt, reference_images_json, status, retry_count,
                started_at, estimated_total_sec, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'running', 0, ?, ?, ?, ?)
            """,
            (
                clip["id"],
                mode,
                requested,
                operator_id,
                operator_name,
                prompt,
                json_text(reference_images or []),
                now,
                estimated_total_sec,
                now,
                now,
            ),
        )
        job_id = cur.lastrowid
        conn.execute("UPDATE clips SET status='generating', updated_at=? WHERE id=?", (now, clip["id"]))
        conn.execute(
            "UPDATE episodes SET final_status='stale', updated_at=? WHERE uuid=? AND final_status IN ('ready','stitching')",
            (now, clip["episode_uuid"]),
        )
    try:
        episode_uuid = clip["episode_uuid"]
        suffix = "dryrun" if dry_run else mode
        out_path = GENERATED_DIR / episode_uuid / f"clip_{int(clip['clip_index']):04d}_job_{job_id}_{suffix}.mp4"
        if dry_run:
            payload = client.dry_run_payload(prompt, clip["public_url"], float(clip["duration_sec"]), reference_images or [])
            out_path = out_path.with_suffix(".json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(__import__("json").dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            task_id = f"dry-run-{job_id}"
            output_url = ""
        elif mode == "mock":
            data = client.mock_generate(Path(clip["local_path"]), out_path)
            task_id = data["task_id"]
            output_url = data["output_url"]
        else:
            key_slot: dict[str, Any] | None = None
            job_for_call = {
                "id": job_id,
                "clip_id": clip["id"],
                "operator_id": operator_id,
                "operator_name": operator_name,
                "requested_duration_sec": requested,
                "duration_sec": clip["duration_sec"],
            }
            try:
                key_slot = acquire_seedance_key_slot(settings)
                client = SeedanceClient({**settings, "seedance_api_key": key_slot["api_key"]})
                job_for_call["api_key_id"] = key_slot["id"]
                job_for_call["api_key_name"] = key_slot["name"]
                try:
                    task = client.create_task(prompt, clip["public_url"], float(clip["duration_sec"]), reference_images or [])
                except Exception as exc:
                    record_seedance_api_call(job_for_call, "failed", error=str(exc))
                    raise
                task_id = task["task_id"]
                job_for_call["task_id"] = task_id
                record_seedance_api_call(
                    job_for_call,
                    "submitted",
                    task_id=task_id,
                    usage=task.get("usage"),
                    raw_response=task.get("raw_response"),
                )
                data = client.wait_for_task(task_id, out_path, input_url=clip["public_url"])
            except Exception as exc:
                if job_for_call.get("task_id"):
                    update_seedance_api_call(job_for_call, "failed", task_id=job_for_call.get("task_id"), error=str(exc))
                raise
            finally:
                release_seedance_key_slot(key_slot)
            update_seedance_api_call(
                job_for_call,
                "succeeded",
                task_id=job_for_call.get("task_id"),
                usage=data.get("usage"),
                raw_response=data.get("raw_response"),
            )
            task_id = data["task_id"]
            output_url = data["output_url"]
        with db.connect() as conn:
            completed_at = db.now()
            conn.execute(
                """
                UPDATE generation_jobs
                SET status='succeeded', task_id=?, output_url=?, output_path=?, error=NULL, completed_at=?, updated_at=?
                WHERE id=?
                """,
                (task_id, output_url, str(out_path.resolve()), completed_at, completed_at, job_id),
            )
            next_clip_status = previous_clip_status if dry_run else "generated"
            conn.execute(
                "UPDATE clips SET status=?, updated_at=? WHERE id=? AND status='generating'",
                (next_clip_status, completed_at, clip["id"]),
            )
        return {"job_id": job_id, "clip_id": clip["id"], "status": "succeeded", "output_path": str(out_path)}
    except Exception as exc:
        fail_generation_job(job_id, int(clip["id"]), str(exc))
        return {"job_id": job_id, "clip_id": clip["id"], "status": "failed", "error": str(exc)}


def retry_job(
    job_id: int,
    lock_token: str | None = None,
    operator_id: str | None = None,
    operator_name: str | None = None,
    prompt: str | None = None,
    reference_images: list[str] | None = None,
) -> dict[str, Any]:
    job = db.one("SELECT * FROM generation_jobs WHERE id=?", (job_id,))
    if not job:
        raise ValueError("job not found")
    clip = db.one("SELECT * FROM clips WHERE id=?", (job["clip_id"],))
    if not clip:
        raise ValueError("clip not found")
    job_reference_images = parse_json_text(job.get("reference_images_json"))
    if not isinstance(job_reference_images, list):
        job_reference_images = None
    return retry_clip(
        clip["id"],
        mode=job["mode"],
        lock_token=lock_token,
        operator_id=operator_id,
        operator_name=operator_name,
        prompt=prompt if prompt is not None else job.get("prompt"),
        reference_images=reference_images if reference_images is not None else job_reference_images,
    )


def retry_clip(
    clip_id: int,
    mode: str | None = None,
    lock_token: str | None = None,
    require_lock_token: bool = True,
    operator_id: str | None = None,
    operator_name: str | None = None,
    prompt: str | None = None,
    reference_images: list[str] | None = None,
) -> dict[str, Any]:
    clip = db.one("SELECT * FROM clips WHERE id=?", (clip_id,))
    if not clip:
        raise ValueError("clip not found")
    if require_lock_token:
        require_lock("episode", clip["episode_uuid"], lock_token)
    deleted_future_clip_count = (
        delete_dependent_continuity_clips(clip)
        if clip.get("input_kind") in CONTINUITY_INPUT_KINDS
        else 0
    )
    tokens = {str(clip["episode_uuid"]): lock_token} if lock_token else {}
    normalized_operator_id, normalized_operator_name = normalize_operator(operator_id, operator_name)
    if not normalized_operator_id:
        normalized_operator_id, normalized_operator_name = operator_from_lock_token(lock_token)
    result = queue_generation(
        clip_ids=[clip_id],
        mode=mode,
        lock_tokens=tokens,
        force=True,
        operator_id=normalized_operator_id,
        operator_name=normalized_operator_name,
        prompt=prompt,
        reference_images=reference_images,
    )[0]
    result["deleted_future_clip_count"] = deleted_future_clip_count
    if clip.get("input_kind") in CONTINUITY_INPUT_KINDS:
        update_continuity_state(clip["episode_uuid"])
    return result


def anchor_stage1_input_path(clip: dict[str, Any]) -> Path:
    return ACCEPTED_DIR / clip["episode_uuid"] / f"clip_{int(clip['clip_index']):04d}_stage1_input.mp4"


def accepted_clip_output_path(clip: dict[str, Any]) -> Path:
    return ACCEPTED_DIR / clip["episode_uuid"] / f"clip_{int(clip['clip_index']):04d}.mp4"


def chronological_accepted_output_path(clip: dict[str, Any]) -> Path:
    return ACCEPTED_DIR / clip["episode_uuid"] / f"clip_{int(clip['clip_index']):04d}_chronological.mp4"


def accepted_output_needs_chronological_sidecar(clip: dict[str, Any]) -> bool:
    return clip.get("input_kind") == "rolling" and str(clip.get("direction") or "forward") == "backward"


def ensure_chronological_accepted_path(
    clip: dict[str, Any],
    accepted_path: Path,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    accepted_path = Path(accepted_path)
    if not accepted_output_needs_chronological_sidecar(clip):
        return accepted_path
    if not accepted_path.exists():
        raise FileNotFoundError(f"accepted output is missing: {accepted_path}")
    chronological_path = chronological_accepted_output_path(clip)
    if accepted_path.resolve() == chronological_path.resolve():
        return chronological_path
    if (
        chronological_path.exists()
        and chronological_path.stat().st_size > 0
        and chronological_path.stat().st_mtime >= accepted_path.stat().st_mtime
    ):
        return chronological_path
    tmp = chronological_path.with_name(
        f".{chronological_path.stem}_{int(time.time() * 1000)}{chronological_path.suffix}"
    )
    try:
        reverse_video(accepted_path, tmp, should_cancel=should_cancel)
        tmp.replace(chronological_path)
    finally:
        tmp.unlink(missing_ok=True)
    return chronological_path


def review_clip(
    clip_id: int,
    decision: str,
    job_id: int | None = None,
    note: str = "",
    lock_token: str | None = None,
    require_lock_token: bool = True,
    operator_id: str | None = None,
    operator_name: str | None = None,
    prompt: str | None = None,
    reference_images: list[str] | None = None,
) -> dict[str, Any]:
    if decision not in {"accept", "reject", "rerun", "flag"}:
        raise ValueError("decision must be accept/reject/rerun/flag")
    clip = db.one("SELECT * FROM clips WHERE id=?", (clip_id,))
    if not clip:
        raise ValueError("clip not found")
    if require_lock_token:
        require_lock("episode", clip["episode_uuid"], lock_token)
    if job_id is None:
        job = db.one(
            "SELECT * FROM generation_jobs WHERE clip_id=? AND status='succeeded' ORDER BY created_at DESC LIMIT 1",
            (clip_id,),
        )
    else:
        job = db.one("SELECT * FROM generation_jobs WHERE id=?", (job_id,))
    if job and job["clip_id"] != clip_id:
        raise ValueError("generation job does not belong to this clip")
    accepted_path = None
    status = {"accept": "accepted", "reject": "rejected", "rerun": "pending", "flag": "flagged"}[decision]
    stage1_anchor_accept = (
        decision == "accept"
        and clip.get("input_kind") == "anchor"
        and clip_anchor_stage(clip) == ANCHOR_STAGE_REPLACE_ARM
    )
    if decision == "accept":
        if not job or not job.get("output_path"):
            raise ValueError("accept requires a succeeded generation job")
        if job.get("status") != "succeeded":
            raise ValueError("accept requires a succeeded generation job")
        if not str(job.get("output_path", "")).lower().endswith(".mp4"):
            raise ValueError("accept requires a generated mp4 output")
        accepted_path_obj = anchor_stage1_input_path(clip) if stage1_anchor_accept else accepted_clip_output_path(clip)
        temp_accepted_path = accepted_path_obj.with_name(
            f".clip_{int(clip['clip_index']):04d}_review_{int(time.time() * 1000)}.mp4"
        )
        try:
            normalize_accepted(Path(job["output_path"]), temp_accepted_path, float(clip["duration_sec"]))
            if require_lock_token:
                require_lock("episode", clip["episode_uuid"], lock_token)
            temp_accepted_path.replace(accepted_path_obj)
            ensure_chronological_accepted_path(clip, accepted_path_obj)
        except Exception:
            temp_accepted_path.unlink(missing_ok=True)
            raise
        accepted_path = str(accepted_path_obj.resolve())
        if stage1_anchor_accept:
            status = "pending"
    elif decision == "rerun":
        rerun_result = retry_clip(
            clip_id,
            lock_token=lock_token,
            require_lock_token=require_lock_token,
            operator_id=operator_id,
            operator_name=operator_name,
            prompt=prompt,
            reference_images=reference_images,
        )
        job = db.one("SELECT * FROM generation_jobs WHERE id=?", (rerun_result["job_id"],)) or job
        if rerun_result.get("status") == "succeeded":
            status = "generated"
        elif rerun_result.get("status") == "queued":
            status = "generating"
        else:
            status = "generated_failed"
    if require_lock_token:
        require_lock("episode", clip["episode_uuid"], lock_token)
    normalized_operator_id, normalized_operator_name = normalize_operator(operator_id, operator_name)
    if not normalized_operator_id:
        normalized_operator_id, normalized_operator_name = operator_from_lock_token(lock_token)
    now = db.now()
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO reviews(clip_id, job_id, operator_id, operator_name, decision, note, accepted_path, reviewed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (clip_id, job["id"] if job else None, normalized_operator_id, normalized_operator_name, decision, note, accepted_path, now),
        )
        if stage1_anchor_accept:
            assert accepted_path is not None
            public_url = public_url_for("accepted", Path(clip["episode_uuid"]) / Path(accepted_path).name)
            conn.execute(
                """
                UPDATE clips
                SET status=?, anchor_stage=?, local_path=?, public_url=?, updated_at=?
                WHERE id=?
                """,
                (status, ANCHOR_STAGE_REPLACE_COLLECTOR, accepted_path, public_url, now, clip_id),
            )
        elif decision == "accept" and clip.get("input_kind") == "anchor":
            conn.execute(
                """
                UPDATE clips
                SET status=?, anchor_stage=?, updated_at=?
                WHERE id=?
                """,
                (status, ANCHOR_STAGE_OFFICIAL, now, clip_id),
            )
        else:
            conn.execute("UPDATE clips SET status=?, updated_at=? WHERE id=?", (status, now, clip_id))
        conn.execute("UPDATE episodes SET final_status='stale', updated_at=? WHERE uuid=?", (now, clip["episode_uuid"]))
    deleted_future_clip_count = (
        delete_dependent_continuity_clips(clip)
        if decision in {"reject", "rerun"} and clip.get("input_kind") in CONTINUITY_INPUT_KINDS
        else 0
    )
    if stage1_anchor_accept:
        next_clip = db.one("SELECT * FROM clips WHERE id=?", (clip_id,))
        next_clips = [next_clip] if next_clip else []
    else:
        next_clips = (
            maybe_prepare_next_continuity_clips_after_accept(clip["episode_uuid"], clip_id, accepted_path)
            if decision == "accept"
            else []
        )
    update_continuity_state(clip["episode_uuid"])
    if stage1_anchor_accept:
        preview = {
            "uuid": clip["episode_uuid"],
            "queued": False,
            "preview_status": db.one("SELECT preview_status FROM episodes WHERE uuid=?", (clip["episode_uuid"],))["preview_status"],
            "reason": "anchor advanced to collector replacement stage",
        }
        final = None
    else:
        preview = queue_preview_episode(clip["episode_uuid"])
        final = maybe_stitch_episode(clip["episode_uuid"])
    return {
        "clip_id": clip_id,
        "decision": decision,
        "accepted_path": accepted_path,
        "next_clip": next_clips[0] if next_clips else None,
        "next_clips": next_clips,
        "deleted_future_clip_count": deleted_future_clip_count,
        "preview": preview,
        "final": final,
    }


def maybe_stitch_episode(uuid: str) -> dict[str, Any] | None:
    clips = db.rows("SELECT * FROM clips WHERE episode_uuid=? ORDER BY clip_index", (uuid,))
    if not clips:
        return None
    if not episode_clips_ready_for_stitch(uuid, clips):
        return None
    return queue_stitch_episode(uuid, check_episode_lock=False)


def queue_preview_episode(uuid: str) -> dict[str, Any]:
    uuid = uuid.lower()
    episode = db.one("SELECT * FROM episodes WHERE uuid=?", (uuid,))
    if not episode:
        return {"uuid": uuid, "queued": False, "preview_status": "missing", "reason": "episode not found"}
    clips = db.rows("SELECT * FROM clips WHERE episode_uuid=? ORDER BY clip_index", (uuid,))
    if not clips:
        with db.connect() as conn:
            conn.execute(
                """
                UPDATE episodes
                SET preview_video_path=NULL, preview_status='missing', preview_error=NULL,
                    preview_version=preview_version+1, updated_at=?
                WHERE uuid=?
                """,
                (db.now(), uuid),
            )
        return {"uuid": uuid, "queued": False, "preview_status": "missing", "reason": "no clips"}
    accepted_count = sum(1 for clip in clips if clip.get("status") == "accepted")
    if accepted_count <= 0:
        with db.connect() as conn:
            conn.execute(
                """
                UPDATE episodes
                SET preview_video_path=NULL, preview_status='missing', preview_error=NULL,
                    preview_version=preview_version+1, updated_at=?
                WHERE uuid=?
                """,
                (db.now(), uuid),
            )
        return {"uuid": uuid, "queued": False, "preview_status": "missing", "reason": "no accepted clips"}
    with db.connect() as conn:
        row = conn.execute("SELECT COALESCE(preview_version, 0) AS version FROM episodes WHERE uuid=?", (uuid,)).fetchone()
        version = int(row["version"] if row else 0) + 1
        conn.execute(
            """
            UPDATE episodes
            SET preview_status='stitching', preview_error=NULL, preview_version=?, updated_at=?
            WHERE uuid=?
            """,
            (version, db.now(), uuid),
        )
    with _PREVIEW_LOCK:
        if uuid not in _PREVIEWING_EPISODES:
            _PREVIEWING_EPISODES.add(uuid)
            _PREVIEW_EXECUTOR.submit(_preview_episode_worker, uuid, version)
    return {"uuid": uuid, "queued": True, "preview_status": "stitching", "preview_version": version}


def _preview_episode_worker(uuid: str, version: int) -> None:
    rerun = False
    try:
        result = preview_episode(uuid, version)
        rerun = bool(result.get("stale"))
    except Exception as exc:
        episode = db.one("SELECT preview_version FROM episodes WHERE uuid=?", (uuid,))
        if episode and int(episode.get("preview_version") or 0) != version:
            rerun = True
        else:
            with db.connect() as conn:
                conn.execute(
                    "UPDATE episodes SET preview_status='failed', preview_error=?, updated_at=? WHERE uuid=?",
                    (str(exc), db.now(), uuid),
                )
    finally:
        with _PREVIEW_LOCK:
            _PREVIEWING_EPISODES.discard(uuid)
            latest = db.one("SELECT preview_status, preview_version FROM episodes WHERE uuid=?", (uuid,))
            latest_still_wants_preview = latest and latest.get("preview_status") == "stitching"
            if latest_still_wants_preview and (rerun or int(latest.get("preview_version") or 0) != version):
                latest_version = int(latest.get("preview_version") or 0) if latest else version + 1
                _PREVIEWING_EPISODES.add(uuid)
                _PREVIEW_EXECUTOR.submit(_preview_episode_worker, uuid, latest_version)


def _preview_is_stale(uuid: str, version: int) -> bool:
    episode = db.one("SELECT preview_version FROM episodes WHERE uuid=?", (uuid,))
    return not episode or int(episode.get("preview_version") or 0) != version


def preview_episode(uuid: str, version: int) -> dict[str, Any]:
    uuid = uuid.lower()
    clips = db.rows("SELECT * FROM clips WHERE episode_uuid=? ORDER BY clip_index", (uuid,))
    if not clips:
        return {"uuid": uuid, "preview_status": "missing", "reason": "no clips"}
    should_cancel = lambda: _preview_is_stale(uuid, version)
    accepted_by_id: dict[int, Path] = {}
    for clip in clips:
        if clip.get("status") != "accepted":
            continue
        review = db.one(
            "SELECT * FROM reviews WHERE clip_id=? AND decision='accept' ORDER BY reviewed_at DESC LIMIT 1",
            (clip["id"],),
        )
        if review and review.get("accepted_path"):
            path = Path(review["accepted_path"])
            if path.exists():
                accepted_by_id[int(clip["id"])] = ensure_chronological_accepted_path(clip, path, should_cancel)
    if _preview_is_stale(uuid, version):
        return {"uuid": uuid, "preview_status": "stitching", "stale": True}
    if not accepted_by_id:
        with db.connect() as conn:
            conn.execute(
                """
                UPDATE episodes
                SET preview_video_path=NULL, preview_status='missing', preview_error=NULL, updated_at=?
                WHERE uuid=? AND preview_version=?
                """,
                (db.now(), uuid, version),
            )
        return {"uuid": uuid, "preview_status": "missing", "reason": "no accepted clips"}
    out = FINAL_DIR / f"{uuid}_preview_accepted_with_black.mp4"
    tmp = FINAL_DIR / f".{uuid}_preview-{version}-{int(time.time() * 1000)}.mp4"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            stitch_inputs = preview_stitch_inputs(uuid, clips, accepted_by_id, tmp_root, should_cancel)
            concat_videos_precise(stitch_inputs, tmp, should_cancel=should_cancel)
        episode = db.one("SELECT preview_version FROM episodes WHERE uuid=?", (uuid,))
        if not episode or int(episode.get("preview_version") or 0) != version:
            tmp.unlink(missing_ok=True)
            return {"uuid": uuid, "preview_status": "stitching", "stale": True}
        tmp.replace(out)
        with db.connect() as conn:
            conn.execute(
                """
                UPDATE episodes
                SET preview_video_path=?, preview_status='ready', preview_error=NULL, updated_at=?
                WHERE uuid=? AND preview_version=?
                """,
                (str(out.resolve()), db.now(), uuid, version),
            )
        return {"uuid": uuid, "preview_video_path": str(out.resolve()), "preview_status": "ready"}
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def preview_stitch_inputs(
    uuid: str,
    clips: list[dict[str, Any]],
    accepted_by_id: dict[int, Path],
    tmp_root: Path,
    should_cancel: Callable[[], bool] | None = None,
) -> list[Path]:
    def check_cancelled() -> None:
        if should_cancel and should_cancel():
            raise RuntimeError("preview cancelled")

    continuity = [clip for clip in clips if clip.get("input_kind") in CONTINUITY_INPUT_KINDS]
    if continuity:
        episode = db.one("SELECT head_video_path FROM episodes WHERE uuid=?", (uuid,))
        total = None
        if episode and episode.get("head_video_path") and Path(episode["head_video_path"]).exists():
            total = continuity_total_duration(Path(episode["head_video_path"]))
        summary = continuity_timeline_summary(uuid, clips, episode)
        relevant = continuity_relevant_clips(
            clips,
            int(summary["anchor_clip_id"]) if summary.get("anchor_clip_id") else None,
        )
        items: list[Path] = []
        cursor = 0.0
        black_index = 0
        for clip in relevant:
            check_cancelled()
            start = clip_timeline_start(clip)
            end = clip_timeline_end(clip)
            if end <= cursor + TIME_EPSILON:
                continue
            if start > cursor + TIME_EPSILON:
                black = tmp_root / f"preview_gap_{black_index:04d}.mp4"
                black_index += 1
                black_video(black, start - cursor, should_cancel=should_cancel)
                items.append(black)
            timeline = max(0.0, end - max(start, cursor))
            if timeline <= TIME_EPSILON:
                cursor = max(cursor, end)
                continue
            if int(clip["id"]) in accepted_by_id:
                accepted_path = accepted_by_id[int(clip["id"])]
                trimmed = tmp_root / f"preview_clip_{int(clip['clip_index']):04d}_trimmed.mp4"
                items.append(trim_continuity_contribution(clip, accepted_path, trimmed, should_cancel=should_cancel))
            else:
                black = tmp_root / f"preview_clip_{int(clip['clip_index']):04d}_black.mp4"
                black_video(black, timeline, should_cancel=should_cancel)
                items.append(black)
            cursor = max(cursor, end)
        if total is not None and total > cursor + TIME_EPSILON:
            black = tmp_root / f"preview_gap_{black_index:04d}.mp4"
            black_video(black, total - cursor, should_cancel=should_cancel)
            items.append(black)
        return items

    items = []
    for clip in clips:
        check_cancelled()
        timeline = float(clip.get("timeline_duration_sec") or clip.get("duration_sec") or 0)
        if timeline <= 0:
            continue
        if int(clip["id"]) in accepted_by_id:
            items.append(accepted_by_id[int(clip["id"])])
        else:
            black = tmp_root / f"preview_clip_{int(clip['clip_index']):04d}_black.mp4"
            black_video(black, timeline, should_cancel=should_cancel)
            items.append(black)
    return items


def episode_clips_ready_for_stitch(uuid: str, clips: list[dict[str, Any]]) -> bool:
    if not clips:
        return False
    continuity = [clip for clip in clips if clip.get("input_kind") in CONTINUITY_INPUT_KINDS]
    if continuity:
        relevant = stitchable_clips(uuid, clips)
        return bool(relevant) and continuity_timeline_summary(uuid, clips).get("complete") is True
    return all(clip["status"] == "accepted" for clip in clips)


def queue_stitch_episode(
    uuid: str,
    lock_token: str | None = None,
    require_lock_token: bool = False,
    check_episode_lock: bool = True,
) -> dict[str, Any]:
    uuid = uuid.lower()
    if check_episode_lock:
        if require_lock_token:
            require_lock("episode", uuid, lock_token)
        else:
            require_no_active_lock("episode", uuid)
    clips = db.rows("SELECT * FROM clips WHERE episode_uuid=? ORDER BY clip_index", (uuid,))
    if not clips:
        return {"uuid": uuid, "queued": False, "final_status": "missing", "reason": "no clips"}
    if not episode_clips_ready_for_stitch(uuid, clips):
        reason = "not all clips accepted" if any(clip["status"] != "accepted" for clip in clips) else "rolling timeline incomplete"
        return {"uuid": uuid, "queued": False, "final_status": "stale", "reason": reason}

    with _STITCH_LOCK:
        if uuid in _STITCHING_EPISODES:
            return {"uuid": uuid, "queued": False, "final_status": "stitching", "reason": "already stitching"}
        stitch_clips = stitchable_clips(uuid, clips)
        lock_tokens = acquire_stitch_locks(uuid, stitch_clips)
        _STITCHING_EPISODES.add(uuid)
        with db.connect() as conn:
            conn.execute("UPDATE episodes SET final_status='stitching', error=NULL, updated_at=? WHERE uuid=?", (db.now(), uuid))
        _STITCH_EXECUTOR.submit(_stitch_episode_worker, uuid, lock_tokens)
    return {"uuid": uuid, "queued": True, "final_status": "stitching"}


def _stitch_episode_worker(uuid: str, lock_tokens: list[str]) -> None:
    try:
        stitch_episode(uuid)
    except Exception as exc:
        with db.connect() as conn:
            conn.execute("UPDATE episodes SET final_status='failed', error=?, updated_at=? WHERE uuid=?", (str(exc), db.now(), uuid))
    finally:
        release_stitch_locks(lock_tokens)
        with _STITCH_LOCK:
            _STITCHING_EPISODES.discard(uuid)


def stitch_episode(uuid: str) -> dict[str, Any]:
    uuid = uuid.lower()
    clips = db.rows("SELECT * FROM clips WHERE episode_uuid=? ORDER BY clip_index", (uuid,))
    stitch_clips = stitchable_clips(uuid, clips)
    accepted_paths = []
    for clip in stitch_clips:
        review = db.one(
            "SELECT * FROM reviews WHERE clip_id=? AND decision='accept' ORDER BY reviewed_at DESC LIMIT 1",
            (clip["id"],),
        )
        if not review or not review.get("accepted_path"):
            raise RuntimeError(f"clip {clip['id']} is not accepted")
        accepted_paths.append(ensure_chronological_accepted_path(clip, Path(review["accepted_path"])))
    out = FINAL_DIR / f"{uuid}_accepted_30fps.mp4"
    tmp = FINAL_DIR / f".{uuid}_accepted_30fps.stitching-{int(time.time() * 1000)}.mp4"
    with db.connect() as conn:
        conn.execute("UPDATE episodes SET final_status='stitching', error=NULL, updated_at=? WHERE uuid=?", (db.now(), uuid))
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            stitch_inputs: list[Path] = []
            uses_precise_concat = False
            for clip, accepted_path in zip(stitch_clips, accepted_paths):
                if clip.get("input_kind") in CONTINUITY_INPUT_KINDS:
                    trimmed = Path(tmpdir) / f"clip_{int(clip['clip_index']):04d}_trimmed.mp4"
                    stitch_inputs.append(trim_continuity_contribution(clip, accepted_path, trimmed))
                    uses_precise_concat = True
                else:
                    stitch_inputs.append(accepted_path)
            if uses_precise_concat:
                concat_videos_precise(stitch_inputs, tmp)
            else:
                stitch_videos(stitch_inputs, tmp)
        episode = db.one("SELECT final_status FROM episodes WHERE uuid=?", (uuid,))
        current_clips = db.rows("SELECT * FROM clips WHERE episode_uuid=?", (uuid,))
        if not episode or episode.get("final_status") != "stitching" or not episode_clips_ready_for_stitch(uuid, current_clips):
            tmp.unlink(missing_ok=True)
            return {"uuid": uuid, "final_status": episode.get("final_status") if episode else "missing", "stale": True}
        tmp.replace(out)
        with db.connect() as conn:
            conn.execute(
                """
                UPDATE episodes
                SET final_video_path=?,
                    final_status='ready',
                    continuity_state=CASE
                        WHEN EXISTS (
                            SELECT 1 FROM clips
                            WHERE episode_uuid=? AND input_kind IN ('anchor','rolling')
                        )
                        THEN 'complete'
                        ELSE continuity_state
                    END,
                    error=NULL,
                    updated_at=?
                WHERE uuid=?
                """,
                (str(out.resolve()), uuid, db.now(), uuid),
            )
        update_continuity_state(uuid)
        return {"uuid": uuid, "final_video_path": str(out.resolve()), "final_status": "ready"}
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        with db.connect() as conn:
            conn.execute("UPDATE episodes SET final_status='failed', error=?, updated_at=? WHERE uuid=?", (str(exc), db.now(), uuid))
        raise


def auto_accept_all(uuid: str | None = None) -> list[dict[str, Any]]:
    if uuid:
        clips = db.rows("SELECT * FROM clips WHERE episode_uuid=? ORDER BY clip_index", (uuid,))
    else:
        clips = db.rows("SELECT * FROM clips ORDER BY episode_uuid, clip_index")
    results = []
    for clip in clips:
        if clip["status"] == "accepted":
            continue
        job = db.one(
            "SELECT * FROM generation_jobs WHERE clip_id=? AND status='succeeded' ORDER BY created_at DESC LIMIT 1",
            (clip["id"],),
        )
        if job:
            results.append(review_clip(clip["id"], "accept", job["id"], "auto-accept", require_lock_token=False))
    return results
