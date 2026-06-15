from __future__ import annotations

import json
import re
import secrets
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlencode

from . import db
from .media import media_url_for_path
from .paths import LAB_DIR, REFERENCE_IMAGES_DIR, ROOT
from .services import (
    acquire_seedance_key_slot,
    generation_overrides,
    json_text,
    normalize_operator,
    parse_json_text,
    release_seedance_key_slot,
)
from .settings import load_settings
from .video import cut_clip, requested_seedance_duration, validate_video_file, video_duration
from ..seedance.client import SeedanceClient


LAB_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".avi"}
LAB_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
LAB_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="seedance-lab")


def _experiment_dir(experiment_id: int) -> Path:
    return LAB_DIR / f"experiment_{int(experiment_id):06d}"


def _job_output_path(experiment_id: int, job_id: int, mode: str) -> Path:
    return _experiment_dir(experiment_id) / "generated" / f"job_{int(job_id):06d}_{mode}.mp4"


def _safe_suffix(filename: str, allowed: set[str], default: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in allowed else default


def _slug(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-")
    return cleaned[:80] or fallback


def _reference_image_id(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _reference_image_payload(ref_id: str) -> dict[str, Any]:
    path = _reference_image_path(ref_id)
    name = path.name if path else Path(ref_id).name
    url = ""
    if path:
        rel = path.resolve().relative_to(REFERENCE_IMAGES_DIR.resolve()).as_posix()
        url = f"/reference_images/{rel}"
    return {"id": ref_id, "name": name, "url": url}


def _reference_image_path(ref_id: str) -> Path | None:
    value = str(ref_id or "").strip()
    if not value:
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    resolved = (ROOT / path).resolve()
    try:
        resolved.relative_to(REFERENCE_IMAGES_DIR.resolve())
    except ValueError:
        return None
    if not resolved.is_file() or resolved.suffix.lower() not in LAB_IMAGE_EXTENSIONS:
        return None
    return resolved


def _validate_reference_images(values: list[str] | None) -> list[str]:
    refs: list[str] = []
    for value in values or []:
        ref_id = str(value or "").strip()
        if not ref_id:
            continue
        if not _reference_image_path(ref_id):
            raise ValueError(f"reference image is not in the project library: {ref_id}")
        refs.append(ref_id)
    return refs


def _parse_json_list(value: Any) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def _api_call_payload(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["usage"] = parse_json_text(item.get("usage_json"))
    item["has_raw_response"] = bool(item.get("raw_response_json"))
    item.pop("usage_json", None)
    item.pop("raw_response_json", None)
    return item


def _default_generation_values() -> tuple[str, list[str]]:
    settings = load_settings()
    return generation_overrides(settings, None, None)


def _normal_mode(value: str | None) -> str:
    mode = (value or "mock").strip().lower()
    if mode not in {"mock", "seedance"}:
        raise ValueError("mode must be mock or seedance")
    return mode


def _clamped_window(source_duration: float, start_sec: float | None, duration_sec: float | None) -> tuple[float, float]:
    if source_duration < 4:
        raise ValueError("reference video must be at least 4 seconds long")
    duration = float(duration_sec) if duration_sec is not None else min(15.0, source_duration)
    duration = max(4.0, min(15.0, duration))
    duration = min(duration, source_duration)
    start = max(0.0, float(start_sec or 0.0))
    if start + duration > source_duration:
        start = max(0.0, source_duration - duration)
    return round(start, 3), round(duration, 3)


def _job_payload(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if not job:
        return None
    result = dict(job)
    now = db.now()
    started = result.get("started_at")
    completed = result.get("completed_at")
    elapsed = max(0.0, float((completed or now) - started)) if started else None
    estimate = result.get("estimated_total_sec")
    result["elapsed_sec"] = elapsed
    result["remaining_estimated_sec"] = (
        max(0.0, float(estimate) - float(elapsed))
        if estimate is not None and elapsed is not None and result.get("status") == "running"
        else None
    )
    result["progress_pct"] = (
        max(1, min(99, int((float(elapsed) / float(estimate)) * 100)))
        if estimate and elapsed is not None and result.get("status") == "running"
        else (100 if result.get("status") == "succeeded" else 0)
    )
    result["generated_url"] = media_url_for_path(result.get("output_path"), LAB_DIR, "lab_generated")
    result["input_video_url"] = media_url_for_path(result.get("input_video_path"), LAB_DIR, "lab_input")
    result["reference_images"] = _parse_json_list(result.get("reference_images_json"))
    result["reference_image_items"] = [_reference_image_payload(ref_id) for ref_id in result["reference_images"]]
    calls = db.rows(
        """
        SELECT * FROM seedance_api_calls
        WHERE lab_job_id=?
        ORDER BY created_at DESC, id DESC
        """,
        (result["id"],),
    )
    result["api_calls"] = [_api_call_payload(call) for call in calls]
    if elapsed is not None and result.get("clip_duration_sec"):
        result["seconds_per_video_second"] = elapsed / float(result["clip_duration_sec"])
    else:
        result["seconds_per_video_second"] = None
    result.pop("reference_images_json", None)
    return result


def _experiment_payload(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    refs = _parse_json_list(item.get("reference_images_json"))
    latest_job = None
    if item.get("latest_job_id"):
        latest_job = db.one("SELECT * FROM lab_generation_jobs WHERE id=?", (item["latest_job_id"],))
    if latest_job is None:
        latest_job = db.one(
            """
            SELECT * FROM lab_generation_jobs
            WHERE experiment_id=?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (item["id"],),
        )
    jobs = db.rows(
        """
        SELECT * FROM lab_generation_jobs
        WHERE experiment_id=?
        ORDER BY created_at DESC, id DESC
        """,
        (item["id"],),
    )
    item["reference_images"] = refs
    item["reference_image_items"] = [_reference_image_payload(ref_id) for ref_id in refs]
    item["source_video_url"] = media_url_for_path(item.get("source_video_path"), LAB_DIR, "lab_source")
    item["input_video_url"] = media_url_for_path(item.get("input_video_path"), LAB_DIR, "lab_input")
    item["latest_job"] = _job_payload(latest_job)
    item["jobs"] = [_job_payload(job) for job in jobs]
    item["generated_url"] = item["latest_job"]["generated_url"] if item.get("latest_job") else None
    item.pop("reference_images_json", None)
    return item


def list_lab_experiments() -> list[dict[str, Any]]:
    rows = db.rows("SELECT * FROM lab_experiments ORDER BY updated_at DESC, id DESC")
    return [_experiment_payload(row) for row in rows]


def get_lab_experiment(experiment_id: int) -> dict[str, Any]:
    row = db.one("SELECT * FROM lab_experiments WHERE id=?", (int(experiment_id),))
    if not row:
        raise ValueError("lab experiment not found")
    return _experiment_payload(row)


def create_lab_experiment(
    title: str | None = None,
    operator_id: str | None = None,
    operator_name: str | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    prompt, refs = _default_generation_values()
    preset_id = str(settings.get("default_generation_preset_id") or "")
    presets = settings.get("generation_presets") if isinstance(settings.get("generation_presets"), list) else []
    preset = next((item for item in presets if str(item.get("id") or "") == preset_id), None)
    preset_name = str(preset.get("name") or "") if preset else ""
    operator_id, operator_name = normalize_operator(operator_id, operator_name)
    now = db.now()
    with db.connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO lab_experiments(
                title, preset_id, preset_name, prompt, reference_images_json, status, mode,
                operator_id, operator_name, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 'draft', 'mock', ?, ?, ?, ?)
            """,
            (
                (title or "").strip() or "新实验",
                preset_id,
                preset_name,
                prompt,
                json_text(refs) or "[]",
                operator_id,
                operator_name,
                now,
                now,
            ),
        )
        experiment_id = int(cur.lastrowid)
        conn.execute(
            "UPDATE lab_experiments SET title=? WHERE id=? AND title='新实验'",
            (f"实验 {experiment_id:04d}", experiment_id),
        )
    _experiment_dir(experiment_id).mkdir(parents=True, exist_ok=True)
    return get_lab_experiment(experiment_id)


def update_lab_experiment(experiment_id: int, values: dict[str, Any]) -> dict[str, Any]:
    current = db.one("SELECT * FROM lab_experiments WHERE id=?", (int(experiment_id),))
    if not current:
        raise ValueError("lab experiment not found")
    updates: dict[str, Any] = {}
    if "title" in values and values["title"] is not None:
        updates["title"] = str(values["title"]).strip() or current["title"]
    if "note" in values and values["note"] is not None:
        updates["note"] = str(values["note"]).strip()
    if "preset_id" in values and values["preset_id"] is not None:
        updates["preset_id"] = str(values["preset_id"]).strip()
    if "preset_name" in values and values["preset_name"] is not None:
        updates["preset_name"] = str(values["preset_name"]).strip()
    if "prompt" in values and values["prompt"] is not None:
        prompt = str(values["prompt"]).strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        updates["prompt"] = prompt
    if "reference_images" in values and values["reference_images"] is not None:
        updates["reference_images_json"] = json_text(_validate_reference_images(values["reference_images"])) or "[]"
    if "mode" in values and values["mode"] is not None:
        updates["mode"] = _normal_mode(values["mode"])
    recut = False
    clip_start = current.get("clip_start_sec")
    clip_duration = current.get("clip_duration_sec")
    if "clip_start_sec" in values and values["clip_start_sec"] is not None:
        clip_start = float(values["clip_start_sec"])
        recut = True
    if "clip_duration_sec" in values and values["clip_duration_sec"] is not None:
        clip_duration = float(values["clip_duration_sec"])
        recut = True
    if recut:
        if not current.get("source_video_path"):
            raise ValueError("upload a reference video before cutting a clip")
        source = Path(current["source_video_path"])
        source_duration = float(current.get("source_duration_sec") or video_duration(source))
        start, duration = _clamped_window(source_duration, clip_start, clip_duration)
        input_path = _experiment_dir(int(experiment_id)) / "input.mp4"
        cut_clip(source, input_path, start, duration)
        validate_video_file(input_path, duration)
        updates["clip_start_sec"] = start
        updates["clip_duration_sec"] = duration
        updates["input_video_path"] = str(input_path.resolve())
        updates["status"] = "ready"
        updates["error"] = None
    if not updates:
        return get_lab_experiment(experiment_id)
    updates["updated_at"] = db.now()
    assignments = ", ".join(f"{key}=?" for key in updates)
    params = [updates[key] for key in updates]
    params.append(int(experiment_id))
    with db.connect() as conn:
        conn.execute(f"UPDATE lab_experiments SET {assignments} WHERE id=?", params)
    return get_lab_experiment(experiment_id)


def save_lab_video_upload(
    experiment_id: int,
    filename: str,
    file_obj: BinaryIO,
    start_sec: float | None = None,
    duration_sec: float | None = None,
) -> dict[str, Any]:
    if not db.one("SELECT id FROM lab_experiments WHERE id=?", (int(experiment_id),)):
        raise ValueError("lab experiment not found")
    suffix = _safe_suffix(filename, LAB_VIDEO_EXTENSIONS, ".mp4")
    directory = _experiment_dir(int(experiment_id))
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / f"source{suffix}"
    with source_path.open("wb") as out:
        shutil.copyfileobj(file_obj, out)
    source_duration = validate_video_file(source_path)
    start, duration = _clamped_window(source_duration, start_sec, duration_sec)
    input_path = directory / "input.mp4"
    cut_clip(source_path, input_path, start, duration)
    validate_video_file(input_path, duration)
    now = db.now()
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE lab_experiments
            SET source_video_path=?, source_video_name=?, source_duration_sec=?, input_video_path=?,
                clip_start_sec=?, clip_duration_sec=?, status='ready',
                latest_job_id=NULL, error=NULL, updated_at=?
            WHERE id=?
            """,
            (
                str(source_path.resolve()),
                filename,
                source_duration,
                str(input_path.resolve()),
                start,
                duration,
                now,
                int(experiment_id),
            ),
        )
    return get_lab_experiment(experiment_id)


def save_lab_reference_images(
    experiment_id: int,
    uploads: list[tuple[str, BinaryIO]],
) -> dict[str, Any]:
    current = db.one("SELECT * FROM lab_experiments WHERE id=?", (int(experiment_id),))
    if not current:
        raise ValueError("lab experiment not found")
    refs = _parse_json_list(current.get("reference_images_json"))
    REFERENCE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for index, (filename, file_obj) in enumerate(uploads, start=1):
        suffix = _safe_suffix(filename, LAB_IMAGE_EXTENSIONS, ".png")
        stem = _slug(Path(filename or "").stem, f"ref-{index}")
        target = REFERENCE_IMAGES_DIR / f"lab-exp-{int(experiment_id):06d}-{int(time.time() * 1000)}-{secrets.token_hex(3)}-{stem}{suffix}"
        with target.open("wb") as out:
            shutil.copyfileobj(file_obj, out)
        refs.append(_reference_image_id(target))
    refs = _validate_reference_images(refs)
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE lab_experiments
            SET reference_images_json=?, updated_at=?
            WHERE id=?
            """,
            (json_text(refs) or "[]", db.now(), int(experiment_id)),
        )
    return get_lab_experiment(experiment_id)


def queue_lab_generation(
    experiment_id: int,
    mode: str | None = None,
    operator_id: str | None = None,
    operator_name: str | None = None,
) -> dict[str, Any]:
    experiment = db.one("SELECT * FROM lab_experiments WHERE id=?", (int(experiment_id),))
    if not experiment:
        raise ValueError("lab experiment not found")
    input_path = Path(str(experiment.get("input_video_path") or ""))
    if not input_path.exists():
        raise ValueError("upload a reference video before running generation")
    mode = _normal_mode(mode or experiment.get("mode"))
    operator_id, operator_name = normalize_operator(operator_id, operator_name)
    duration = float(experiment.get("clip_duration_sec") or video_duration(input_path))
    requested = requested_seedance_duration(duration)
    if requested < 4 or requested > 15:
        raise ValueError("Seedance input duration must be between 4 and 15 seconds")
    settings = load_settings()
    estimate = (
        max(0.1, duration * float(settings.get("mock_seconds_per_video_second") or 0.25))
        if mode == "mock"
        else requested * float(settings.get("seedance_seconds_per_video_second") or 24)
    )
    refs = _validate_reference_images(_parse_json_list(experiment.get("reference_images_json")))
    now = db.now()
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        active = conn.execute(
            """
            SELECT id FROM lab_generation_jobs
            WHERE experiment_id=? AND status IN ('queued','running')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (int(experiment_id),),
        ).fetchone()
        if active:
            return get_lab_experiment(experiment_id)
        cur = conn.execute(
            """
            INSERT INTO lab_generation_jobs(
                experiment_id, mode, requested_duration_sec, operator_id, operator_name,
                preset_id, preset_name, prompt, reference_images_json,
                source_video_path, source_video_name, input_video_path,
                clip_start_sec, clip_duration_sec, source_duration_sec,
                status, estimated_total_sec, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)
            """,
            (
                int(experiment_id),
                mode,
                requested,
                operator_id,
                operator_name,
                str(experiment.get("preset_id") or ""),
                str(experiment.get("preset_name") or ""),
                str(experiment.get("prompt") or ""),
                json_text(refs) or "[]",
                str(experiment.get("source_video_path") or ""),
                str(experiment.get("source_video_name") or ""),
                str(input_path.resolve()),
                float(experiment.get("clip_start_sec") or 0.0),
                duration,
                float(experiment.get("source_duration_sec") or 0.0),
                estimate,
                now,
                now,
            ),
        )
        job_id = int(cur.lastrowid)
        conn.execute(
            """
            UPDATE lab_experiments
            SET status='generating', mode=?, latest_job_id=?, error=NULL,
                operator_id=?, operator_name=?, updated_at=?
            WHERE id=?
            """,
            (mode, job_id, operator_id, operator_name, now, int(experiment_id)),
        )
    LAB_EXECUTOR.submit(_lab_generation_worker, job_id)
    return get_lab_experiment(experiment_id)


def _lab_generation_worker(job_id: int) -> None:
    job = db.one(
        """
        SELECT j.*, COALESCE(j.input_video_path, e.input_video_path) AS input_video_path,
               COALESCE(j.clip_duration_sec, e.clip_duration_sec) AS clip_duration_sec
        FROM lab_generation_jobs j
        JOIN lab_experiments e ON e.id = j.experiment_id
        WHERE j.id=?
        """,
        (int(job_id),),
    )
    if not job or job.get("status") not in {"queued", "running"}:
        return
    now = db.now()
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE lab_generation_jobs
            SET status='running', started_at=COALESCE(started_at, ?), error=NULL, updated_at=?
            WHERE id=? AND status IN ('queued','running')
            """,
            (now, now, int(job_id)),
        )
    job["status"] = "running"
    output_path = _job_output_path(int(job["experiment_id"]), int(job_id), str(job["mode"]))
    try:
        input_path = Path(str(job["input_video_path"]))
        duration = float(job.get("clip_duration_sec") or video_duration(input_path))
        validate_video_file(input_path, duration)
        if job["mode"] == "mock":
            remaining = max(0.0, float(job.get("estimated_total_sec") or 0))
            while remaining > 0:
                sleep_for = min(0.5, remaining)
                time.sleep(sleep_for)
                remaining -= sleep_for
                with db.connect() as conn:
                    conn.execute(
                        "UPDATE lab_generation_jobs SET updated_at=? WHERE id=? AND status='running'",
                        (db.now(), int(job_id)),
                    )
            data = SeedanceClient(load_settings()).mock_generate(input_path, output_path)
            _complete_lab_generation(job, data["task_id"], data.get("output_url") or "", output_path)
            return
        _run_lab_seedance_job(job, input_path, output_path, duration)
    except Exception as exc:
        _fail_lab_generation(int(job_id), int(job["experiment_id"]), str(exc))


def _lab_input_url(input_path: Path, job_id: int) -> str:
    media_url = media_url_for_path(input_path, LAB_DIR, "lab_seedance_input", absolute=True)
    if not media_url:
        raise RuntimeError("failed to issue media token for lab input video")
    params = urlencode({"lab_job": int(job_id), "v": int(db.now() * 1000)})
    return f"{media_url}{'&' if '?' in media_url else '?'}{params}"


def _run_lab_seedance_job(job: dict[str, Any], input_path: Path, output_path: Path, duration: float) -> None:
    settings = load_settings()
    key_slot: dict[str, Any] | None = None
    call_record = {
        "lab_job_id": int(job["id"]),
        "operator_id": job.get("operator_id") or "",
        "operator_name": job.get("operator_name") or "",
        "requested_duration_sec": job.get("requested_duration_sec"),
        "clip_duration_sec": duration,
        "api_key_id": "",
        "api_key_name": "",
    }
    try:
        key_slot = acquire_seedance_key_slot(settings)
        call_record["api_key_id"] = key_slot["id"]
        call_record["api_key_name"] = key_slot["name"]
        client = SeedanceClient({**settings, "seedance_api_key": key_slot["api_key"]})
        input_url = _lab_input_url(input_path, int(job["id"]))
        refs = _parse_json_list(job.get("reference_images_json"))
        try:
            task = client.create_task(str(job.get("prompt") or ""), input_url, duration, refs)
        except Exception as exc:
            _record_lab_seedance_call(call_record, "failed", error=str(exc))
            raise
        task_id = task["task_id"]
        call_record["task_id"] = task_id
        _record_lab_seedance_call(
            call_record,
            "submitted",
            task_id=task_id,
            usage=task.get("usage"),
            raw_response=task.get("raw_response"),
        )
        with db.connect() as conn:
            conn.execute("UPDATE lab_generation_jobs SET task_id=?, updated_at=? WHERE id=?", (task_id, db.now(), job["id"]))
        data = client.wait_for_task(
            task_id,
            output_path,
            input_url=input_url,
            on_poll=lambda _task: _lab_job_heartbeat(int(job["id"])),
            on_download_progress=lambda _received, _expected: _lab_job_heartbeat(int(job["id"])),
        )
        _update_lab_seedance_call(
            call_record,
            "succeeded",
            task_id=task_id,
            usage=data.get("usage"),
            raw_response=data.get("raw_response"),
        )
        _complete_lab_generation(job, task_id, data.get("output_url") or "", output_path)
    except Exception as exc:
        if call_record.get("task_id"):
            _update_lab_seedance_call(call_record, "failed", task_id=call_record.get("task_id"), error=str(exc))
        raise
    finally:
        release_seedance_key_slot(key_slot)


def _lab_job_heartbeat(job_id: int) -> None:
    with db.connect() as conn:
        conn.execute(
            "UPDATE lab_generation_jobs SET updated_at=? WHERE id=? AND status='running'",
            (db.now(), int(job_id)),
        )


def _complete_lab_generation(job: dict[str, Any], task_id: str, output_url: str, output_path: Path) -> None:
    now = db.now()
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE lab_generation_jobs
            SET status='succeeded', task_id=?, output_url=?, output_path=?,
                error=NULL, completed_at=?, updated_at=?
            WHERE id=?
            """,
            (task_id, output_url, str(output_path.resolve()), now, now, int(job["id"])),
        )
        conn.execute(
            """
            UPDATE lab_experiments
            SET status='generated', latest_job_id=?, error=NULL, updated_at=?
            WHERE id=?
            """,
            (int(job["id"]), now, int(job["experiment_id"])),
        )


def _fail_lab_generation(job_id: int, experiment_id: int, error: str) -> None:
    now = db.now()
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE lab_generation_jobs
            SET status='failed', error=?, completed_at=?, updated_at=?
            WHERE id=?
            """,
            (error, now, now, int(job_id)),
        )
        conn.execute(
            """
            UPDATE lab_experiments
            SET status='generated_failed', error=?, updated_at=?
            WHERE id=?
            """,
            (error, now, int(experiment_id)),
        )


def _seedance_call_values(
    job: dict[str, Any],
    status: str,
    task_id: str | None = None,
    usage: Any = None,
    raw_response: Any = None,
    error: str | None = None,
) -> dict[str, Any]:
    now = db.now()
    return {
        "job_id": None,
        "clip_id": None,
        "lab_job_id": job.get("lab_job_id"),
        "operator_id": job.get("operator_id") or "",
        "operator_name": job.get("operator_name") or "",
        "api_key_id": job.get("api_key_id") or "",
        "api_key_name": job.get("api_key_name") or "",
        "call_type": "lab_create_task",
        "status": status,
        "task_id": task_id or job.get("task_id") or "",
        "model": load_settings().get("seedance_model") or "",
        "requested_duration_sec": job.get("requested_duration_sec"),
        "clip_duration_sec": job.get("clip_duration_sec"),
        "usage_json": json_text(usage),
        "raw_response_json": json_text(raw_response),
        "error": error,
        "created_at": now,
        "updated_at": now,
    }


def _record_lab_seedance_call(
    job: dict[str, Any],
    status: str,
    task_id: str | None = None,
    usage: Any = None,
    raw_response: Any = None,
    error: str | None = None,
) -> None:
    payload = _seedance_call_values(job, status, task_id, usage, raw_response, error)
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO seedance_api_calls(
                job_id, clip_id, lab_job_id, operator_id, operator_name,
                api_key_id, api_key_name, call_type, status, task_id, model,
                requested_duration_sec, clip_duration_sec, usage_json,
                raw_response_json, error, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["job_id"],
                payload["clip_id"],
                payload["lab_job_id"],
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


def _update_lab_seedance_call(
    job: dict[str, Any],
    status: str,
    task_id: str | None = None,
    usage: Any = None,
    raw_response: Any = None,
    error: str | None = None,
) -> None:
    row = db.one(
        """
        SELECT id FROM seedance_api_calls
        WHERE lab_job_id=? AND call_type='lab_create_task'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (job.get("lab_job_id"),),
    )
    if not row:
        _record_lab_seedance_call(job, status, task_id, usage, raw_response, error)
        return
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE seedance_api_calls
            SET status=?, task_id=?, usage_json=COALESCE(?, usage_json),
                raw_response_json=COALESCE(?, raw_response_json), error=?, updated_at=?
            WHERE id=?
            """,
            (status, task_id or job.get("task_id") or "", json_text(usage), json_text(raw_response), error, db.now(), row["id"]),
        )
