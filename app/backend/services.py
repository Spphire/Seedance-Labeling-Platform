from __future__ import annotations

import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from . import db
from .ids import parse_uuids
from .locks import LockError, active_lock, list_locks, locks_by_resource, require_lock, require_no_active_lock
from .nedf import extract_head_video, fetch_episode
from .paths import ACCEPTED_DIR, CLIPS_DIR, EPISODES_DIR, FINAL_DIR, GENERATED_DIR, HEAD_VIDEOS_DIR
from .settings import load_settings, public_url_for
from .video import clip_plan, cut_clip, normalize_accepted, requested_seedance_duration, stitch_videos, transcode_760x570, video_duration
from ..seedance.client import SeedanceClient


_STITCH_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="seedance-stitch")
_STITCH_LOCK = threading.Lock()
_STITCHING_EPISODES: set[str] = set()


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
        episode["review_remaining_count"] = max(clip_count - accepted, 0)
        episode["manual_decision_count"] = generated + flagged + rejected
        episode["episode_stage"] = describe_episode_stage(episode)
        episode["lock"] = episode_locks.get(episode["uuid"])
        final_path = episode.get("final_video_path")
        episode["final_url"] = static_url_from_path(final_path, FINAL_DIR, "final") if final_path else None
    return episodes


def describe_episode_stage(episode: dict[str, Any]) -> str:
    clip_count = int(episode.get("clip_count") or 0)
    accepted = int(episode.get("accepted_clip_count") or 0)
    final_status = episode.get("final_status") or "missing"
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
                WHERE status='succeeded'
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
        clip["generated_url"] = (
            static_url_from_path(latest_job.get("output_path"), GENERATED_DIR, "generated") if latest_job else None
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
    return jobs


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
        clip_lock = active_clip_lock_for_episode(uuid)
        if clip_lock:
            raise LockError(409, "episode has an active clip lock", clip_lock)
        return
    raise LockError(423, "episode mutation requires an active episode lock", episode_lock)


def active_clip_lock_for_episode(uuid: str) -> dict[str, Any] | None:
    clip_ids = {str(row["id"]) for row in db.rows("SELECT id FROM clips WHERE episode_uuid=?", (uuid,))}
    if not clip_ids:
        return None
    for lock in list_locks():
        if lock["resource_type"] == "clip" and lock["resource_id"] in clip_ids:
            return lock
    return None


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


def preprocess_one(uuid: str, settings: dict[str, Any], fetch_remote: bool, lock_token: str | None = None) -> dict[str, Any]:
    now = db.now()
    local_dir = EPISODES_DIR / uuid
    require_episode_mutation_lock(uuid, lock_token)
    with db.connect() as conn:
        conn.execute("UPDATE episodes SET status='preprocessing', error=NULL, updated_at=? WHERE uuid=?", (now, uuid))
    try:
        if fetch_remote:
            fetch_episode(settings["dm3_host"], settings["dm3_nedf_root"], uuid, local_dir)
        preprocessed_dir = local_dir / "preprocessed"
        if not (preprocessed_dir / "metadata.json").exists():
            raise RuntimeError(f"Missing preprocessed metadata for {uuid}")
        head_path = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        meta = extract_head_video(preprocessed_dir, head_path)
        duration = video_duration(head_path)
        clips = create_clips(uuid, head_path, duration)
        with db.connect() as conn:
            conn.execute(
                """
                UPDATE episodes SET status='preprocessed', head_video_path=?, final_status='missing',
                error=NULL, updated_at=? WHERE uuid=?
                """,
                (str(head_path.resolve()), db.now(), uuid),
            )
        return {"uuid": uuid, "head": meta, "clips": clips}
    except Exception as exc:
        with db.connect() as conn:
            conn.execute("UPDATE episodes SET status='failed', error=?, updated_at=? WHERE uuid=?", (str(exc), db.now(), uuid))
        return {"uuid": uuid, "error": str(exc)}


def create_clips(uuid: str, head_path: Path, duration: float) -> list[dict[str, Any]]:
    plan = clip_plan(duration)
    with db.connect() as conn:
        conn.execute(
            "DELETE FROM resource_locks WHERE resource_type='clip' AND resource_id IN (SELECT CAST(id AS TEXT) FROM clips WHERE episode_uuid=?)",
            (uuid,),
        )
        conn.execute("DELETE FROM reviews WHERE clip_id IN (SELECT id FROM clips WHERE episode_uuid=?)", (uuid,))
        conn.execute("DELETE FROM generation_jobs WHERE clip_id IN (SELECT id FROM clips WHERE episode_uuid=?)", (uuid,))
        conn.execute("DELETE FROM clips WHERE episode_uuid=?", (uuid,))
    clip_dir = CLIPS_DIR / uuid
    if clip_dir.exists():
        shutil.rmtree(clip_dir)
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
                INSERT INTO clips(episode_uuid, clip_index, start_sec, duration_sec, local_path, public_url, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (uuid, index, start, clip_duration, str(path.resolve()), public_url, now, now),
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
    clips = create_clips(uuid, head_path, duration)
    return {"uuid": uuid, "head_video_path": str(head_path.resolve()), "duration_sec": duration, "clips": clips}


def run_generation(
    mode: str | None = None,
    clip_ids: list[int] | None = None,
    dry_run: bool = False,
    lock_tokens: dict[str, str] | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    settings = load_settings()
    mode = mode or settings["generation_mode"]
    if mode not in {"mock", "seedance"}:
        raise ValueError("mode must be mock or seedance")
    if clip_ids is not None and len(clip_ids) == 0:
        return []
    if clip_ids is not None:
        clips = db.rows("SELECT * FROM clips WHERE id IN (%s)" % ",".join("?" for _ in clip_ids), clip_ids)
    else:
        clips = db.rows("SELECT * FROM clips WHERE status IN ('pending','generated_failed') ORDER BY episode_uuid, clip_index")
    clips = filter_generation_clips(clips, lock_tokens or {}, strict=clip_ids is not None)
    if not clips:
        return []
    client = SeedanceClient(settings)
    concurrency_key = "mock_concurrency" if mode == "mock" or dry_run else "seedance_concurrency"
    max_workers = max(1, int(settings.get(concurrency_key, 1)))
    max_workers = min(max_workers, len(clips))
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_generation_for_clip, client, clip, mode, settings, dry_run, force) for clip in clips]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def filter_generation_clips(
    clips: list[dict[str, Any]],
    lock_tokens: dict[str, str],
    strict: bool,
) -> list[dict[str, Any]]:
    active_clip_locks = locks_by_resource("clip")
    available = []
    for clip in clips:
        clip_id = str(clip["id"])
        if clip_id not in active_clip_locks:
            available.append(clip)
            continue
        token = lock_tokens.get(clip_id)
        if token:
            require_lock("clip", clip_id, token)
            available.append(clip)
        elif strict:
            require_lock("clip", clip_id, None)
    return available


def run_generation_for_clip(
    client: SeedanceClient,
    clip: dict[str, Any],
    mode: str,
    settings: dict[str, Any],
    dry_run: bool,
    force: bool = False,
) -> dict[str, Any]:
    requested = requested_seedance_duration(float(clip["duration_sec"]))
    now = db.now()
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute("SELECT status FROM clips WHERE id=?", (clip["id"],)).fetchone()
        if not current:
            return {"clip_id": clip["id"], "status": "skipped", "reason": "clip not found"}
        can_claim = current["status"] != "generating" if force else current["status"] in {"pending", "generated_failed"}
        if not can_claim:
            return {
                "clip_id": clip["id"],
                "status": "skipped",
                "reason": f"clip status is {current['status']}",
            }
        cur = conn.execute(
            """
            INSERT INTO generation_jobs(clip_id, mode, requested_duration_sec, status, retry_count, created_at, updated_at)
            VALUES (?, ?, ?, 'running', 0, ?, ?)
            """,
            (clip["id"], mode, requested, now, now),
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
            payload = client.dry_run_payload(settings["default_prompt"], clip["public_url"], float(clip["duration_sec"]))
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
            data = client.generate(settings["default_prompt"], clip["public_url"], float(clip["duration_sec"]), out_path)
            task_id = data["task_id"]
            output_url = data["output_url"]
        with db.connect() as conn:
            conn.execute(
                """
                UPDATE generation_jobs SET status='succeeded', task_id=?, output_url=?, output_path=?, error=NULL, updated_at=?
                WHERE id=?
                """,
                (task_id, output_url, str(out_path.resolve()), db.now(), job_id),
            )
            conn.execute("UPDATE clips SET status='generated', updated_at=? WHERE id=? AND status='generating'", (db.now(), clip["id"]))
        return {"job_id": job_id, "clip_id": clip["id"], "status": "succeeded", "output_path": str(out_path)}
    except Exception as exc:
        with db.connect() as conn:
            conn.execute("UPDATE generation_jobs SET status='failed', error=?, updated_at=? WHERE id=?", (str(exc), db.now(), job_id))
            conn.execute("UPDATE clips SET status='generated_failed', updated_at=? WHERE id=? AND status='generating'", (db.now(), clip["id"]))
        return {"job_id": job_id, "clip_id": clip["id"], "status": "failed", "error": str(exc)}


def retry_job(job_id: int, lock_token: str | None = None) -> dict[str, Any]:
    job = db.one("SELECT * FROM generation_jobs WHERE id=?", (job_id,))
    if not job:
        raise ValueError("job not found")
    clip = db.one("SELECT * FROM clips WHERE id=?", (job["clip_id"],))
    if not clip:
        raise ValueError("clip not found")
    return retry_clip(clip["id"], mode=job["mode"], lock_token=lock_token)


def retry_clip(clip_id: int, mode: str | None = None, lock_token: str | None = None, require_lock_token: bool = True) -> dict[str, Any]:
    clip = db.one("SELECT * FROM clips WHERE id=?", (clip_id,))
    if not clip:
        raise ValueError("clip not found")
    if require_lock_token:
        require_lock("clip", clip_id, lock_token)
    tokens = {str(clip_id): lock_token} if lock_token else {}
    return run_generation(clip_ids=[clip_id], mode=mode, lock_tokens=tokens, force=True)[0]


def review_clip(
    clip_id: int,
    decision: str,
    job_id: int | None = None,
    note: str = "",
    lock_token: str | None = None,
    require_lock_token: bool = True,
) -> dict[str, Any]:
    if decision not in {"accept", "reject", "rerun", "flag"}:
        raise ValueError("decision must be accept/reject/rerun/flag")
    if require_lock_token:
        require_lock("clip", clip_id, lock_token)
    clip = db.one("SELECT * FROM clips WHERE id=?", (clip_id,))
    if not clip:
        raise ValueError("clip not found")
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
                require_lock("clip", clip_id, lock_token)
            temp_accepted_path.replace(accepted_path_obj)
        except Exception:
            temp_accepted_path.unlink(missing_ok=True)
            raise
        accepted_path = str(accepted_path_obj.resolve())
    elif decision == "rerun":
        rerun_result = retry_clip(clip_id, lock_token=lock_token, require_lock_token=require_lock_token)
        job = db.one("SELECT * FROM generation_jobs WHERE id=?", (rerun_result["job_id"],)) or job
        status = "generated" if rerun_result.get("status") == "succeeded" else "generated_failed"
    if require_lock_token:
        require_lock("clip", clip_id, lock_token)
    now = db.now()
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO reviews(clip_id, job_id, decision, note, accepted_path, reviewed_at) VALUES (?, ?, ?, ?, ?, ?)",
            (clip_id, job["id"] if job else None, decision, note, accepted_path, now),
        )
        conn.execute("UPDATE clips SET status=?, updated_at=? WHERE id=?", (status, now, clip_id))
        conn.execute("UPDATE episodes SET final_status='stale', updated_at=? WHERE uuid=?", (now, clip["episode_uuid"]))
    final = maybe_stitch_episode(clip["episode_uuid"])
    return {"clip_id": clip_id, "decision": decision, "accepted_path": accepted_path, "final": final}


def maybe_stitch_episode(uuid: str) -> dict[str, Any] | None:
    clips = db.rows("SELECT * FROM clips WHERE episode_uuid=? ORDER BY clip_index", (uuid,))
    if not clips:
        return None
    if any(clip["status"] != "accepted" for clip in clips):
        return None
    return queue_stitch_episode(uuid, check_episode_lock=False)


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
    clips = db.rows("SELECT id, status FROM clips WHERE episode_uuid=? ORDER BY clip_index", (uuid,))
    if not clips:
        return {"uuid": uuid, "queued": False, "final_status": "missing", "reason": "no clips"}
    if any(clip["status"] != "accepted" for clip in clips):
        return {"uuid": uuid, "queued": False, "final_status": "stale", "reason": "not all clips accepted"}

    with _STITCH_LOCK:
        if uuid in _STITCHING_EPISODES:
            return {"uuid": uuid, "queued": False, "final_status": "stitching", "reason": "already stitching"}
        _STITCHING_EPISODES.add(uuid)
        with db.connect() as conn:
            conn.execute("UPDATE episodes SET final_status='stitching', error=NULL, updated_at=? WHERE uuid=?", (db.now(), uuid))
        _STITCH_EXECUTOR.submit(_stitch_episode_worker, uuid)
    return {"uuid": uuid, "queued": True, "final_status": "stitching"}


def _stitch_episode_worker(uuid: str) -> None:
    try:
        stitch_episode(uuid)
    except Exception as exc:
        with db.connect() as conn:
            conn.execute("UPDATE episodes SET final_status='failed', error=?, updated_at=? WHERE uuid=?", (str(exc), db.now(), uuid))
    finally:
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
        stitch_videos(accepted_paths, tmp)
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
