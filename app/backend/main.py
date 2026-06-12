from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import db
from .auth import dev_mode_enabled, require_admin, require_reviewer, security_enabled
from .locks import LockError, acquire_lock, list_locks, release_lock, renew_lock
from .media import resolve_media_token
from .paths import ACCEPTED_DIR, CLIPS_DIR, FINAL_DIR, GENERATED_DIR, HEAD_VIDEOS_DIR, REFERENCE_IMAGES_DIR, ROOT, ensure_dirs
from .schema import (
    AnchorCandidatesRequest,
    EpisodeBatchRequest,
    GenerationRunRequest,
    ImportHeadVideoRequest,
    LabExperimentCreateRequest,
    LabExperimentUpdateRequest,
    LabGenerationRequest,
    LockReleaseRequest,
    LockRenewRequest,
    LockRequest,
    LockTokenRequest,
    PreprocessRequest,
    ReviewRequest,
    SubmitPreprocessRequest,
)
from .lab import (
    create_lab_experiment,
    get_lab_experiment,
    list_lab_experiments,
    queue_lab_generation,
    save_lab_reference_images,
    save_lab_video_upload,
    update_lab_experiment,
)
from .services import (
    auto_accept_all,
    create_anchor_candidates,
    import_head_video,
    list_clips,
    list_episodes,
    list_jobs,
    list_reviewer_activity,
    list_seedance_usage,
    preprocess,
    queue_generation,
    queue_rolling_generation,
    queue_stitch_episode,
    recover_interrupted_generation_jobs,
    refresh_clip_public_urls,
    retry_clip,
    retry_job,
    review_clip,
    start_generation_watchdog,
    submit_and_preprocess_episodes,
    submit_episodes,
)
from .settings import load_settings, public_settings, save_settings


FRONTEND_DIR = ROOT / "app" / "frontend"


app = FastAPI(title="Seedance Labeling Platform", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: dict[str, Any]) -> FileResponse:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


def _init() -> None:
    ensure_dirs()
    db.init_db()
    load_settings()


@app.on_event("startup")
def startup() -> None:
    _init()
    refresh_clip_public_urls()
    recover_interrupted_generation_jobs(include_download_failures=True)
    start_generation_watchdog()


def _public_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, LockError):
        return HTTPException(status_code=exc.status_code, detail=exc.detail())
    return HTTPException(status_code=400, detail=str(exc))


def _media_response(path: Path, media_type: str, request: Request) -> Response:
    file_size = path.stat().st_size
    range_header = request.headers.get("range")
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600"}
    if not range_header:
        return FileResponse(path, media_type=media_type, headers=headers)
    unit, _, value = range_header.partition("=")
    if unit.strip().lower() != "bytes" or "-" not in value:
        raise HTTPException(status_code=416, detail="invalid range")
    start_text, _, end_text = value.partition("-")
    try:
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else file_size - 1
    except ValueError as exc:
        raise HTTPException(status_code=416, detail="invalid range") from exc
    start = max(0, start)
    end = min(file_size - 1, end)
    if start > end or start >= file_size:
        raise HTTPException(status_code=416, detail="range not satisfiable")

    def iter_file() -> Any:
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers |= {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Content-Length": str(end - start + 1),
    }
    return StreamingResponse(iter_file(), status_code=206, media_type=media_type, headers=headers)


def frontend_index() -> FileResponse:
    _init()
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend index.html not found")
    return FileResponse(index_path)


@app.get("/")
def index() -> FileResponse:
    return frontend_index()


@app.get("/label")
def label_index() -> FileResponse:
    return frontend_index()


@app.get("/admin")
def admin_index() -> FileResponse:
    return frontend_index()


@app.get("/lab")
def lab_index() -> FileResponse:
    return frontend_index()


@app.get("/api/health")
def health() -> dict[str, Any]:
    _init()
    return {"ok": True, "auth_required": security_enabled(), "settings": public_settings()}


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    return public_settings()


@app.post("/api/settings")
def post_settings(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    require_admin(request)
    values = payload.get("values", payload)
    save_settings(values)
    if "public_base_url" in values:
        refresh_clip_public_urls()
    return public_settings()


@app.post("/api/episodes/batch")
def post_episodes_batch(payload: EpisodeBatchRequest, request: Request) -> list[dict[str, Any]]:
    require_reviewer(request)
    try:
        return submit_episodes(payload.episodes_text)
    except Exception as exc:
        raise _public_error(exc) from exc


@app.post("/api/pipeline/submit_preprocess")
def post_submit_preprocess(payload: SubmitPreprocessRequest, request: Request) -> dict[str, Any]:
    require_reviewer(request)
    try:
        return submit_and_preprocess_episodes(payload.episodes_text, payload.fetch_remote, payload.lock_tokens)
    except Exception as exc:
        raise _public_error(exc) from exc


@app.post("/api/pipeline/preprocess")
def post_preprocess(payload: PreprocessRequest, request: Request) -> list[dict[str, Any]]:
    require_reviewer(request)
    try:
        return preprocess(payload.uuids, payload.fetch_remote, payload.lock_tokens)
    except Exception as exc:
        raise _public_error(exc) from exc


@app.post("/api/pipeline/import_head")
def post_import_head(payload: ImportHeadVideoRequest, request: Request) -> dict[str, Any]:
    require_reviewer(request)
    try:
        return import_head_video(payload.uuid, payload.path, payload.lock_token)
    except Exception as exc:
        raise _public_error(exc) from exc


@app.post("/api/episodes/{uuid}/anchor_candidates")
def post_anchor_candidates(uuid: str, payload: AnchorCandidatesRequest, request: Request) -> dict[str, Any]:
    require_reviewer(request)
    try:
        return create_anchor_candidates(uuid.lower(), payload.start_secs, payload.lock_token)
    except Exception as exc:
        raise _public_error(exc) from exc


@app.post("/api/generation/run")
def post_generation_run(payload: GenerationRunRequest, request: Request) -> list[dict[str, Any]]:
    require_reviewer(request)
    try:
        lock_tokens = {}
        if payload.clip_ids and payload.lock_token:
            lock_tokens = {str(clip_id): payload.lock_token for clip_id in payload.clip_ids}
            placeholders = ",".join("?" for _ in payload.clip_ids)
            rows = db.rows(f"SELECT id, episode_uuid FROM clips WHERE id IN ({placeholders})", payload.clip_ids)
            lock_tokens.update({str(row["episode_uuid"]): payload.lock_token for row in rows})
        if payload.lock_tokens:
            lock_tokens.update({str(key): str(value) for key, value in payload.lock_tokens.items()})
        return queue_generation(
            payload.mode,
            payload.clip_ids,
            payload.dry_run,
            lock_tokens,
            operator_id=payload.operator_id,
            operator_name=payload.operator_name,
            prompt=payload.prompt,
            reference_images=payload.reference_images,
        )
    except Exception as exc:
        raise _public_error(exc) from exc


@app.post("/api/generation/rolling_run")
def post_generation_rolling_run(payload: GenerationRunRequest, request: Request) -> list[dict[str, Any]]:
    require_reviewer(request)
    try:
        return queue_rolling_generation(
            payload.mode,
            payload.dry_run,
            payload.lock_tokens,
            operator_id=payload.operator_id,
            operator_name=payload.operator_name,
            prompt=payload.prompt,
            reference_images=payload.reference_images,
        )
    except Exception as exc:
        raise _public_error(exc) from exc


@app.post("/api/generation/{job_id}/retry")
def post_generation_retry(job_id: int, request: Request, payload: LockTokenRequest | None = None) -> dict[str, Any]:
    require_reviewer(request)
    try:
        return retry_job(
            job_id,
            payload.lock_token if payload else None,
            payload.operator_id if payload else None,
            payload.operator_name if payload else None,
            payload.prompt if payload else None,
            payload.reference_images if payload else None,
        )
    except Exception as exc:
        raise _public_error(exc) from exc


@app.post("/api/clips/{clip_id}/retry")
def post_clip_retry(clip_id: int, request: Request, payload: LockTokenRequest | None = None) -> dict[str, Any]:
    require_reviewer(request)
    try:
        return retry_clip(
            clip_id,
            mode=payload.mode if payload else None,
            lock_token=payload.lock_token if payload else None,
            operator_id=payload.operator_id if payload else None,
            operator_name=payload.operator_name if payload else None,
            prompt=payload.prompt if payload else None,
            reference_images=payload.reference_images if payload else None,
        )
    except Exception as exc:
        raise _public_error(exc) from exc


@app.post("/api/review/{clip_id}")
def post_review(clip_id: int, payload: ReviewRequest, request: Request) -> dict[str, Any]:
    require_reviewer(request)
    try:
        return review_clip(
            clip_id,
            payload.decision,
            payload.job_id,
            payload.note,
            payload.lock_token,
            operator_id=payload.operator_id,
            operator_name=payload.operator_name,
            prompt=payload.prompt,
            reference_images=payload.reference_images,
        )
    except Exception as exc:
        raise _public_error(exc) from exc


@app.post("/api/episodes/{uuid}/stitch")
def post_stitch(uuid: str, request: Request, payload: LockTokenRequest | None = None) -> dict[str, Any]:
    require_reviewer(request)
    try:
        return queue_stitch_episode(uuid.lower(), payload.lock_token if payload else None, require_lock_token=True)
    except Exception as exc:
        raise _public_error(exc) from exc


@app.post("/api/test/auto_accept")
def post_auto_accept(request: Request, payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if not dev_mode_enabled():
        raise HTTPException(status_code=404, detail="not found")
    require_admin(request)
    try:
        uuid = None
        if payload:
            uuid = payload.get("uuid")
        return auto_accept_all(uuid.lower() if isinstance(uuid, str) and uuid else None)
    except Exception as exc:
        raise _public_error(exc) from exc


@app.get("/api/episodes")
def get_episodes() -> list[dict[str, Any]]:
    return list_episodes()


@app.get("/api/clips")
def get_clips() -> list[dict[str, Any]]:
    return list_clips()


@app.get("/api/jobs")
def get_jobs() -> list[dict[str, Any]]:
    return list_jobs()


@app.get("/api/lab/experiments")
def get_lab_experiments_api(request: Request) -> list[dict[str, Any]]:
    require_reviewer(request)
    return list_lab_experiments()


@app.post("/api/lab/experiments")
def post_lab_experiment(payload: LabExperimentCreateRequest, request: Request) -> dict[str, Any]:
    require_reviewer(request)
    try:
        return create_lab_experiment(payload.title, payload.operator_id, payload.operator_name)
    except Exception as exc:
        raise _public_error(exc) from exc


@app.get("/api/lab/experiments/{experiment_id}")
def get_lab_experiment_api(experiment_id: int, request: Request) -> dict[str, Any]:
    require_reviewer(request)
    try:
        return get_lab_experiment(experiment_id)
    except Exception as exc:
        raise _public_error(exc) from exc


@app.patch("/api/lab/experiments/{experiment_id}")
def patch_lab_experiment(experiment_id: int, payload: LabExperimentUpdateRequest, request: Request) -> dict[str, Any]:
    require_reviewer(request)
    try:
        return update_lab_experiment(experiment_id, payload.model_dump(exclude_unset=True))
    except Exception as exc:
        raise _public_error(exc) from exc


@app.post("/api/lab/experiments/{experiment_id}/video")
async def post_lab_video(
    experiment_id: int,
    request: Request,
    file: UploadFile = File(...),
    start_sec: float | None = Form(None),
    duration_sec: float | None = Form(None),
) -> dict[str, Any]:
    require_reviewer(request)
    try:
        return save_lab_video_upload(experiment_id, file.filename or "source.mp4", file.file, start_sec, duration_sec)
    except Exception as exc:
        raise _public_error(exc) from exc
    finally:
        await file.close()


@app.post("/api/lab/experiments/{experiment_id}/images")
async def post_lab_images(
    experiment_id: int,
    request: Request,
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    require_reviewer(request)
    try:
        return save_lab_reference_images(experiment_id, [(file.filename or "ref.png", file.file) for file in files])
    except Exception as exc:
        raise _public_error(exc) from exc
    finally:
        for file in files:
            await file.close()


@app.post("/api/lab/experiments/{experiment_id}/run")
def post_lab_run(experiment_id: int, payload: LabGenerationRequest, request: Request) -> dict[str, Any]:
    require_reviewer(request)
    try:
        return queue_lab_generation(experiment_id, payload.mode, payload.operator_id, payload.operator_name)
    except Exception as exc:
        raise _public_error(exc) from exc


@app.get("/api/usage/seedance")
def get_seedance_usage(request: Request, limit: int = 100) -> dict[str, Any]:
    require_admin(request)
    return list_seedance_usage(limit)


@app.get("/api/reviews/activity")
def get_reviewer_activity(request: Request, limit: int = 100) -> dict[str, Any]:
    require_admin(request)
    return list_reviewer_activity(limit)


@app.get("/api/locks")
def get_locks(request: Request) -> list[dict[str, Any]]:
    require_admin(request)
    return list_locks()


@app.post("/api/locks/acquire")
def post_lock_acquire(payload: LockRequest, request: Request) -> dict[str, Any]:
    require_reviewer(request)
    try:
        return acquire_lock(
            payload.resource_type,
            payload.resource_id,
            payload.owner_id,
            payload.owner_name,
            payload.ttl_sec,
            payload.force,
        )
    except Exception as exc:
        raise _public_error(exc) from exc


@app.post("/api/locks/renew")
def post_lock_renew(payload: LockRenewRequest, request: Request) -> dict[str, Any]:
    require_reviewer(request)
    try:
        return renew_lock(payload.token, payload.owner_id, payload.ttl_sec)
    except Exception as exc:
        raise _public_error(exc) from exc


@app.post("/api/locks/release")
def post_lock_release(payload: LockReleaseRequest, request: Request) -> dict[str, Any]:
    require_reviewer(request)
    try:
        return release_lock(payload.token, payload.owner_id)
    except Exception as exc:
        raise _public_error(exc) from exc


@app.get("/api/state")
def get_state(request: Request) -> dict[str, Any]:
    require_reviewer(request)
    return {
        "episodes": list_episodes(),
        "clips": list_clips(),
        "jobs": list_jobs(),
        "locks": list_locks(),
        "settings": public_settings(),
    }


@app.get("/media/{token}")
def get_media(token: str, request: Request) -> Response:
    row = resolve_media_token(token)
    if not row:
        raise HTTPException(status_code=404, detail="media not found")
    return _media_response(row["path_obj"], row["media_type"], request)


def _mount_static(prefix: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    app.mount(prefix, NoCacheStaticFiles(directory=directory), name=prefix.strip("/"))


if not security_enabled() or dev_mode_enabled():
    _mount_static("/clips", CLIPS_DIR)
    _mount_static("/head_videos", HEAD_VIDEOS_DIR)
    _mount_static("/generated", GENERATED_DIR)
    _mount_static("/accepted", ACCEPTED_DIR)
    _mount_static("/final", FINAL_DIR)
_mount_static("/reference_images", REFERENCE_IMAGES_DIR)
_mount_static("/assets", FRONTEND_DIR)
