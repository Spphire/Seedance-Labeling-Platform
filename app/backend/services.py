from __future__ import annotations

import json
import secrets
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from . import db
from .ids import parse_uuids
from .locks import LockError, active_lock, locks_by_resource, require_lock, require_no_active_lock
from .nedf import extract_head_video, fetch_episode
from .paths import ACCEPTED_DIR, CLIPS_DIR, EPISODES_DIR, FINAL_DIR, GENERATED_DIR, HEAD_VIDEOS_DIR
from .settings import load_settings, public_url_for
from .video import (
    clip_plan,
    concat_videos_precise,
    compose_rolling_input,
    cut_clip,
    normalize_accepted,
    requested_seedance_duration,
    rolling_clip_plan,
    stitch_videos,
    transcode_760x570,
    trim_video,
    video_duration,
)
from ..seedance.client import SeedanceClient


_STITCH_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="seedance-stitch")
_STITCH_LOCK = threading.Lock()
_STITCHING_EPISODES: set[str] = set()
STITCH_LOCK_OWNER_ID = "system-stitcher"
STITCH_LOCK_OWNER_NAME = "系统合成"
STITCH_LOCK_TTL_SEC = 60 * 60 * 24
_GENERATION_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="seedance-generation")
_GENERATION_CONDITION = threading.Condition()
_GENERATION_ACTIVE = 0
GENERATION_CANDIDATE_STATUSES = ("pending", "generated_failed", "rejected")
GENERATION_CANDIDATE_STATUS_PLACEHOLDERS = ",".join("?" for _ in GENERATION_CANDIDATE_STATUSES)
DEFAULT_REQUEST_MODE = "mock"


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
    if episode.get("preprocess_health") == "damaged":
        return "预处理文件疑似损坏，需要重新预处理"
    if clip_count == 0 and episode.get("status") == "preprocessed":
        return "head 已就绪，等待滚动生成"
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
    rolling_clips = [clip for clip in clips if clip.get("input_kind") == "rolling"]
    planned_sec = None
    planned_clip_count = None
    plan_error = ""
    head_value = episode.get("head_video_path")
    if head_value and Path(head_value).exists():
        try:
            plan = rolling_clip_plan(video_duration(Path(head_value)))
            planned_sec = sum(float(item["timeline_duration_sec"]) for item in plan)
            planned_clip_count = len(plan)
        except Exception as exc:
            plan_error = str(exc)
    accepted_sec = sum(
        float(clip.get("timeline_duration_sec") or clip.get("duration_sec") or 0)
        for clip in rolling_clips
        if clip.get("status") == "accepted"
    )
    complete = bool(
        rolling_clips
        and planned_clip_count is not None
        and len(rolling_clips) == planned_clip_count
        and all(clip.get("status") == "accepted" for clip in rolling_clips)
    )
    return {
        "rolling_clip_count": len(rolling_clips),
        "legacy_clip_count": len([clip for clip in clips if clip.get("input_kind") != "rolling"]),
        "rolling_planned_sec": planned_sec,
        "rolling_planned_clip_count": planned_clip_count,
        "rolling_accepted_sec": accepted_sec,
        "rolling_remaining_sec": max(0.0, float(planned_sec) - accepted_sec) if planned_sec is not None else None,
        "rolling_complete": complete,
        "rolling_plan_error": plan_error,
    }


def list_clips() -> list[dict[str, Any]]:
    clips = db.rows(
        """
        SELECT c.*, e.final_status
        FROM clips c
        JOIN episodes e ON e.uuid = c.episode_uuid
        ORDER BY c.episode_uuid, c.clip_index
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
        clip["video_url"] = static_url_from_path(clip.get("local_path"), CLIPS_DIR, "clips")
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
    for call in calls:
        call["usage"] = parse_json_text(call.get("usage_json"))
        call.pop("usage_json", None)
        call.pop("raw_response_json", None)
    return {"summary": summary_rows, "recent_calls": calls}


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
        episode = db.one("SELECT head_video_path FROM episodes WHERE uuid=?", (uuid,))
        first_clip = (
            ensure_initial_rolling_clip(uuid, Path(episode["head_video_path"]))
            if episode and episode.get("head_video_path")
            else None
        )
        return {
            "uuid": uuid,
            "status": "skipped",
            "reason": integrity["reason"],
            "duration_sec": integrity.get("duration_sec"),
            "clip_count": 1 if first_clip else 0,
            "clips": [first_clip] if first_clip else [],
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
            with db.connect() as conn:
                conn.execute(
                    """
                    UPDATE episodes SET status='preprocessed', head_video_path=?, final_status='missing',
                    error=NULL, updated_at=? WHERE uuid=?
                    """,
                    (str(existing_head_path.resolve()), db.now(), uuid),
                )
            first_clip = ensure_initial_rolling_clip(uuid, existing_head_path)
            return {
                "uuid": uuid,
                "status": "preprocessed",
                "reason": integrity["reason"],
                "head": {"output_path": str(existing_head_path.resolve()), "duration_sec": duration, "reused": True},
                "clips": [first_clip] if first_clip else [],
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
                local_path=?, error=NULL, updated_at=? WHERE uuid=?
                """,
                (str(head_path.resolve()), str(episode_dir.resolve()), db.now(), uuid),
            )
        first_clip = ensure_initial_rolling_clip(uuid, head_path)
        return {"uuid": uuid, "status": "preprocessed", "reason": integrity["reason"], "head": meta, "clips": [first_clip] if first_clip else []}
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
    for directory in [CLIPS_DIR / uuid, GENERATED_DIR / uuid, ACCEPTED_DIR / uuid]:
        if directory.exists():
            shutil.rmtree(directory)


def delete_rolling_clips_after(uuid: str, clip_index: int) -> int:
    clips = db.rows(
        "SELECT * FROM clips WHERE episode_uuid=? AND input_kind='rolling' AND clip_index>? ORDER BY clip_index",
        (uuid, clip_index),
    )
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
        generated_dir = GENERATED_DIR / uuid
        if generated_dir.exists():
            for path in generated_dir.glob(f"clip_{int(clip['clip_index']):04d}_job_*"):
                path.unlink(missing_ok=True)
    return len(clips)


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
    first_clip = ensure_initial_rolling_clip(uuid, head_path)
    return {
        "uuid": uuid,
        "head_video_path": str(head_path.resolve()),
        "duration_sec": duration,
        "clips": [first_clip] if first_clip else [],
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
    episodes = db.rows(
        """
        SELECT * FROM episodes
        WHERE status='preprocessed' AND head_video_path IS NOT NULL
        ORDER BY created_at
        """
    )
    prepared: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for episode in episodes:
        result = prepare_rolling_clip_for_episode(episode)
        if result.get("clip"):
            prepared.append(result["clip"])
        else:
            skipped.append(result)
    return prepared, skipped


def prepare_rolling_clip_for_episode(episode: dict[str, Any]) -> dict[str, Any]:
    uuid = episode["uuid"]
    head_value = episode.get("head_video_path")
    if not head_value:
        return rolling_skip(uuid, "head video path missing")
    head_path = Path(head_value)
    if not head_path.exists():
        return rolling_skip(uuid, "head video file missing")
    try:
        plan = rolling_clip_plan(video_duration(head_path))
    except Exception as exc:
        return rolling_skip(uuid, str(exc))
    if not plan:
        return rolling_skip(uuid, "rolling plan is empty")

    clips = db.rows("SELECT * FROM clips WHERE episode_uuid=? ORDER BY clip_index", (uuid,))
    legacy = [clip for clip in clips if clip.get("input_kind") != "rolling"]
    if legacy:
        return rolling_skip(uuid, "legacy split clips exist; rolling generation skipped")
    if clips:
        last = clips[-1]
        if last["status"] in GENERATION_CANDIDATE_STATUSES:
            try:
                ensure_rolling_clip_input(last)
            except Exception as exc:
                return rolling_skip(uuid, f"cannot rebuild rolling input for clip {last['id']}: {exc}")
            return {"episode_uuid": uuid, "clip": db.one("SELECT * FROM clips WHERE id=?", (last["id"],)) or last}
        if last["status"] in {"generated", "flagged", "generating", "preparing"}:
            return rolling_skip(uuid, f"waiting for clip {last['id']} status {last['status']}")
        if last["status"] != "accepted":
            return rolling_skip(uuid, f"clip {last['id']} status is {last['status']}")
        if len(clips) >= len(plan):
            final = maybe_stitch_episode(uuid)
            return {
                "episode_uuid": uuid,
                "status": "skipped",
                "reason": "rolling episode complete",
                "final": final,
            }

    next_index = len(clips)
    plan_item = plan[next_index]
    previous_clip = clips[-1] if clips else None
    previous_accepted_path = None
    if previous_clip:
        previous_accepted_path = latest_accepted_path(int(previous_clip["id"]))
        if not previous_accepted_path:
            return rolling_skip(uuid, f"accepted anchor missing for clip {previous_clip['id']}")
    clip = insert_rolling_clip(uuid, plan_item)
    try:
        build_rolling_clip_input(clip, previous_accepted_path)
    except Exception as exc:
        with db.connect() as conn:
            conn.execute("UPDATE clips SET status='generated_failed', updated_at=? WHERE id=?", (db.now(), clip["id"]))
        return rolling_skip(uuid, f"cannot build rolling input for clip {clip['id']}: {exc}")
    with db.connect() as conn:
        conn.execute("UPDATE clips SET status='pending', updated_at=? WHERE id=?", (db.now(), clip["id"]))
    return {"episode_uuid": uuid, "clip": db.one("SELECT * FROM clips WHERE id=?", (clip["id"],)) or clip}


def ensure_initial_rolling_clip(uuid: str, head_path: Path) -> dict[str, Any] | None:
    plan = rolling_clip_plan(video_duration(head_path))
    if not plan:
        return None
    clips = db.rows("SELECT * FROM clips WHERE episode_uuid=? ORDER BY clip_index", (uuid,))
    if any(clip.get("input_kind") != "rolling" for clip in clips):
        return None
    if clips:
        return clips[0]
    clip = insert_rolling_clip(uuid, plan[0])
    try:
        build_rolling_clip_input(clip, None)
    except Exception:
        with db.connect() as conn:
            conn.execute("UPDATE clips SET status='generated_failed', updated_at=? WHERE id=?", (db.now(), clip["id"]))
        raise
    with db.connect() as conn:
        conn.execute("UPDATE clips SET status='pending', updated_at=? WHERE id=?", (db.now(), clip["id"]))
    return db.one("SELECT * FROM clips WHERE id=?", (clip["id"],)) or clip


def maybe_prepare_next_rolling_clip_after_accept(uuid: str, accepted_clip_id: int, accepted_path: str | None) -> dict[str, Any] | None:
    if not accepted_path:
        return None
    accepted_clip = db.one("SELECT * FROM clips WHERE id=?", (accepted_clip_id,))
    if not accepted_clip or accepted_clip.get("input_kind") != "rolling":
        return None
    accepted_index = int(accepted_clip["clip_index"])
    later = db.one(
        "SELECT * FROM clips WHERE episode_uuid=? AND clip_index>? ORDER BY clip_index LIMIT 1",
        (uuid, accepted_index),
    )
    if later:
        return None
    episode = db.one("SELECT * FROM episodes WHERE uuid=?", (uuid,))
    if not episode or not episode.get("head_video_path"):
        return None
    head_path = Path(episode["head_video_path"])
    if not head_path.exists():
        return None
    plan = rolling_clip_plan(video_duration(head_path))
    next_index = accepted_index + 1
    if next_index >= len(plan):
        return None
    clip = insert_rolling_clip(uuid, plan[next_index])
    try:
        build_rolling_clip_input(clip, Path(accepted_path))
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


def insert_rolling_clip(uuid: str, plan_item: dict[str, Any]) -> dict[str, Any]:
    clip_index = int(plan_item["clip_index"])
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
                source_start_sec, source_duration_sec, overlap_sec, timeline_duration_sec, input_kind,
                local_path, public_url, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'rolling', ?, ?, 'preparing', ?, ?)
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


def ensure_rolling_clip_input(clip: dict[str, Any]) -> None:
    path = Path(clip["local_path"])
    if path.exists():
        return
    previous_accepted_path = None
    if float(clip.get("overlap_sec") or 0) > 0:
        previous = db.one(
            "SELECT * FROM clips WHERE episode_uuid=? AND clip_index=? AND input_kind='rolling'",
            (clip["episode_uuid"], int(clip["clip_index"]) - 1),
        )
        if not previous:
            raise RuntimeError("previous rolling clip is missing")
        previous_accepted_path = latest_accepted_path(int(previous["id"]))
        if not previous_accepted_path:
            raise RuntimeError("previous accepted output is missing")
    build_rolling_clip_input(clip, previous_accepted_path)


def build_rolling_clip_input(clip: dict[str, Any], previous_accepted_path: Path | None) -> None:
    episode = db.one("SELECT * FROM episodes WHERE uuid=?", (clip["episode_uuid"],))
    if not episode or not episode.get("head_video_path"):
        raise RuntimeError("episode head video path missing")
    head_path = Path(episode["head_video_path"])
    if not head_path.exists():
        raise RuntimeError("episode head video file missing")
    compose_rolling_input(
        head_path,
        Path(clip["local_path"]),
        float(clip.get("source_start_sec") if clip.get("source_start_sec") is not None else clip["start_sec"]),
        float(clip.get("source_duration_sec") if clip.get("source_duration_sec") is not None else clip["duration_sec"]),
        previous_accepted_path,
        float(clip.get("overlap_sec") or 0),
    )


def latest_accepted_path(clip_id: int) -> Path | None:
    review = db.one(
        "SELECT * FROM reviews WHERE clip_id=? AND decision='accept' ORDER BY reviewed_at DESC LIMIT 1",
        (clip_id,),
    )
    if not review or not review.get("accepted_path"):
        return None
    path = Path(review["accepted_path"])
    return path if path.exists() else None


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
    available = filter_generation_clips(clips, {}, strict=False)
    if not available:
        return []
    operator_id, operator_name = normalize_operator(operator_id, operator_name)
    generation_prompt, generation_refs = generation_overrides(settings, prompt, reference_images)
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
                    generation_prompt,
                    generation_refs,
                )
                for clip in available
            ]
            for future in as_completed(futures):
                results.append(future.result())
        return results

    claimed = []
    for clip in available:
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
    clips = filter_generation_clips(clips, lock_tokens or {}, strict=clip_ids is not None)
    if not clips:
        return []
    client = SeedanceClient(settings)
    operator_id, operator_name = normalize_operator(operator_id, operator_name)
    generation_prompt, generation_refs = generation_overrides(settings, prompt, reference_images)
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
                generation_prompt,
                generation_refs,
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
    clips = filter_generation_clips(clips, lock_tokens or {}, strict=clip_ids is not None)
    if not clips:
        return []
    claimed = []
    operator_id, operator_name = normalize_operator(operator_id, operator_name)
    generation_prompt, generation_refs = generation_overrides(settings, prompt, reference_images)
    for clip in clips:
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
    job = db.one(
        """
        SELECT j.*, c.episode_uuid, c.clip_index, c.duration_sec, c.public_url
        FROM generation_jobs j
        JOIN clips c ON c.id = j.clip_id
        WHERE j.id=?
        """,
        (job_id,),
    )
    if not job or job.get("status") != "running":
        return
    job = dict(job)
    settings = load_settings()
    client = SeedanceClient(settings)
    job_prompt, job_refs = generation_values_from_job(job, settings)
    out_path = GENERATED_DIR / job["episode_uuid"] / f"clip_{int(job['clip_index']):04d}_job_{job_id}_seedance.mp4"
    max_workers = max(1, int(settings.get("seedance_concurrency", 1)))
    acquire_generation_slot(max_workers)
    try:
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
        data = client.wait_for_task(job["task_id"], out_path, input_url=job["public_url"])
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
        if job.get("task_id"):
            update_seedance_api_call(job, "failed", task_id=job.get("task_id"), error=str(exc))
        fail_generation_job(job_id, int(job["clip_id"]), str(exc))
    finally:
        release_generation_slot()


def acquire_generation_slot(max_workers: int) -> None:
    global _GENERATION_ACTIVE
    with _GENERATION_CONDITION:
        while _GENERATION_ACTIVE >= max_workers:
            _GENERATION_CONDITION.wait(timeout=5)
        _GENERATION_ACTIVE += 1


def release_generation_slot() -> None:
    global _GENERATION_ACTIVE
    with _GENERATION_CONDITION:
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


def filter_generation_clips(
    clips: list[dict[str, Any]],
    lock_tokens: dict[str, str],
    strict: bool,
) -> list[dict[str, Any]]:
    active_clip_locks = locks_by_resource("clip")
    available = []
    for clip in clips:
        clip_id = str(clip["id"])
        clip_token = lock_tokens.get(clip_id)
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
                job_id, clip_id, operator_id, operator_name, call_type, status,
                task_id, model, requested_duration_sec, clip_duration_sec,
                usage_json, raw_response_json, error, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["job_id"],
                payload["clip_id"],
                payload["operator_id"],
                payload["operator_name"],
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
            job_for_call = {
                "id": job_id,
                "clip_id": clip["id"],
                "operator_id": operator_id,
                "operator_name": operator_name,
                "requested_duration_sec": requested,
                "duration_sec": clip["duration_sec"],
            }
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
            try:
                data = client.wait_for_task(task_id, out_path, input_url=clip["public_url"])
            except Exception as exc:
                update_seedance_api_call(job_for_call, "failed", task_id=task_id, error=str(exc))
                raise
            update_seedance_api_call(
                job_for_call,
                "succeeded",
                task_id=task_id,
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
    tokens = {str(clip["episode_uuid"]): lock_token} if lock_token else {}
    normalized_operator_id, normalized_operator_name = normalize_operator(operator_id, operator_name)
    if not normalized_operator_id:
        normalized_operator_id, normalized_operator_name = operator_from_lock_token(lock_token)
    return queue_generation(
        clip_ids=[clip_id],
        mode=mode,
        lock_tokens=tokens,
        force=True,
        operator_id=normalized_operator_id,
        operator_name=normalized_operator_name,
        prompt=prompt,
        reference_images=reference_images,
    )[0]


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
    if decision == "accept":
        if not job or not job.get("output_path"):
            raise ValueError("accept requires a succeeded generation job")
        if job.get("status") != "succeeded":
            raise ValueError("accept requires a succeeded generation job")
        if not str(job.get("output_path", "")).lower().endswith(".mp4"):
            raise ValueError("accept requires a generated mp4 output")
        accepted_path_obj = ACCEPTED_DIR / clip["episode_uuid"] / f"clip_{int(clip['clip_index']):04d}.mp4"
        temp_accepted_path = accepted_path_obj.with_name(
            f".clip_{int(clip['clip_index']):04d}_review_{int(time.time() * 1000)}.mp4"
        )
        try:
            normalize_accepted(Path(job["output_path"]), temp_accepted_path, float(clip["duration_sec"]))
            if require_lock_token:
                require_lock("episode", clip["episode_uuid"], lock_token)
            temp_accepted_path.replace(accepted_path_obj)
        except Exception:
            temp_accepted_path.unlink(missing_ok=True)
            raise
        accepted_path = str(accepted_path_obj.resolve())
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
        conn.execute("UPDATE clips SET status=?, updated_at=? WHERE id=?", (status, now, clip_id))
        conn.execute("UPDATE episodes SET final_status='stale', updated_at=? WHERE uuid=?", (now, clip["episode_uuid"]))
    deleted_future_clip_count = (
        delete_rolling_clips_after(clip["episode_uuid"], int(clip["clip_index"]))
        if decision in {"reject", "rerun"} and clip.get("input_kind") == "rolling"
        else 0
    )
    next_clip = (
        maybe_prepare_next_rolling_clip_after_accept(clip["episode_uuid"], clip_id, accepted_path)
        if decision == "accept"
        else None
    )
    final = maybe_stitch_episode(clip["episode_uuid"])
    return {
        "clip_id": clip_id,
        "decision": decision,
        "accepted_path": accepted_path,
        "next_clip": next_clip,
        "deleted_future_clip_count": deleted_future_clip_count,
        "final": final,
    }


def maybe_stitch_episode(uuid: str) -> dict[str, Any] | None:
    clips = db.rows("SELECT * FROM clips WHERE episode_uuid=? ORDER BY clip_index", (uuid,))
    if not clips:
        return None
    if not episode_clips_ready_for_stitch(uuid, clips):
        return None
    return queue_stitch_episode(uuid, check_episode_lock=False)


def episode_clips_ready_for_stitch(uuid: str, clips: list[dict[str, Any]]) -> bool:
    if not clips:
        return False
    if any(clip["status"] != "accepted" for clip in clips):
        return False
    rolling_count = len([clip for clip in clips if clip.get("input_kind") == "rolling"])
    if rolling_count == 0:
        return True
    if rolling_count != len(clips):
        return False
    episode = db.one("SELECT head_video_path FROM episodes WHERE uuid=?", (uuid,))
    if not episode or not episode.get("head_video_path"):
        return False
    head_path = Path(episode["head_video_path"])
    if not head_path.exists():
        return False
    try:
        plan = rolling_clip_plan(video_duration(head_path))
    except Exception:
        return False
    return len(clips) == len(plan)


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
        lock_tokens = acquire_stitch_locks(uuid, clips)
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
    accepted_paths = []
    for clip in clips:
        review = db.one(
            "SELECT * FROM reviews WHERE clip_id=? AND decision='accept' ORDER BY reviewed_at DESC LIMIT 1",
            (clip["id"],),
        )
        if not review or not review.get("accepted_path"):
            raise RuntimeError(f"clip {clip['id']} is not accepted")
        accepted_paths.append(Path(review["accepted_path"]))
    out = FINAL_DIR / f"{uuid}_accepted_30fps.mp4"
    tmp = FINAL_DIR / f".{uuid}_accepted_30fps.stitching-{int(time.time() * 1000)}.mp4"
    with db.connect() as conn:
        conn.execute("UPDATE episodes SET final_status='stitching', error=NULL, updated_at=? WHERE uuid=?", (db.now(), uuid))
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            stitch_inputs: list[Path] = []
            uses_rolling_trim = False
            for clip, accepted_path in zip(clips, accepted_paths):
                overlap = float(clip.get("overlap_sec") or 0)
                if clip.get("input_kind") == "rolling" and overlap > 0:
                    trimmed = Path(tmpdir) / f"clip_{int(clip['clip_index']):04d}_trimmed.mp4"
                    trim_video(accepted_path, trimmed, overlap, float(clip.get("timeline_duration_sec") or clip["duration_sec"]))
                    stitch_inputs.append(trimmed)
                    uses_rolling_trim = True
                else:
                    stitch_inputs.append(accepted_path)
            if uses_rolling_trim:
                concat_videos_precise(stitch_inputs, tmp)
            else:
                stitch_videos(stitch_inputs, tmp)
        episode = db.one("SELECT final_status FROM episodes WHERE uuid=?", (uuid,))
        current_clips = db.rows("SELECT status FROM clips WHERE episode_uuid=?", (uuid,))
        if not episode or episode.get("final_status") != "stitching" or any(clip["status"] != "accepted" for clip in current_clips):
            tmp.unlink(missing_ok=True)
            return {"uuid": uuid, "final_status": episode.get("final_status") if episode else "missing", "stale": True}
        tmp.replace(out)
        with db.connect() as conn:
            conn.execute(
                "UPDATE episodes SET final_video_path=?, final_status='ready', error=NULL, updated_at=? WHERE uuid=?",
                (str(out.resolve()), db.now(), uuid),
            )
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
