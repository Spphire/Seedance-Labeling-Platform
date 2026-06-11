from __future__ import annotations

import mimetypes
import secrets
from pathlib import Path
from typing import Any

from . import db
from .paths import ACCEPTED_DIR, CLIPS_DIR, FINAL_DIR, GENERATED_DIR, HEAD_VIDEOS_DIR
from .settings import load_settings


MEDIA_TOKEN_TTL_SEC = 24 * 60 * 60
MEDIA_TOKEN_BYTES = 24
MEDIA_ROOTS: dict[str, Path] = {
    "clips": CLIPS_DIR,
    "accepted": ACCEPTED_DIR,
    "generated": GENERATED_DIR,
    "final": FINAL_DIR,
    "head_videos": HEAD_VIDEOS_DIR,
}


def _path_under_root(path_value: str | Path, root: Path) -> Path | None:
    try:
        path = Path(path_value).resolve()
        path.relative_to(root.resolve())
        return path
    except (OSError, ValueError):
        return None


def cleanup_expired_media_tokens(now: float | None = None) -> None:
    with db.connect() as conn:
        conn.execute("DELETE FROM media_tokens WHERE expires_at <= ?", (now or db.now(),))


def issue_media_token(path_value: str | Path, root: Path, purpose: str, ttl_sec: int = MEDIA_TOKEN_TTL_SEC) -> str | None:
    path = _path_under_root(path_value, root)
    if not path or not path.exists() or not path.is_file():
        return None
    now = db.now()
    cleanup_expired_media_tokens(now)
    token = secrets.token_urlsafe(MEDIA_TOKEN_BYTES)
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO media_tokens(token, path, purpose, expires_at, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (token, str(path), purpose, now + ttl_sec, now),
        )
    return token


def media_url_for_path(
    path_value: str | Path | None,
    root: Path,
    purpose: str,
    *,
    absolute: bool = False,
    ttl_sec: int = MEDIA_TOKEN_TTL_SEC,
) -> str | None:
    if not path_value:
        return None
    token = issue_media_token(path_value, root, purpose, ttl_sec)
    if not token:
        return None
    path = f"/media/{token}"
    if not absolute:
        return path
    base = str(load_settings()["public_base_url"]).rstrip("/")
    return f"{base}{path}"


def media_url_for_known_roots(
    path_value: str | Path | None,
    purpose: str,
    *,
    absolute: bool = False,
    ttl_sec: int = MEDIA_TOKEN_TTL_SEC,
) -> str | None:
    if not path_value:
        return None
    for root in MEDIA_ROOTS.values():
        url = media_url_for_path(path_value, root, purpose, absolute=absolute, ttl_sec=ttl_sec)
        if url:
            return url
    return None


def resolve_media_token(token: str) -> dict[str, Any] | None:
    if not token:
        return None
    now = db.now()
    cleanup_expired_media_tokens(now)
    row = db.one("SELECT * FROM media_tokens WHERE token=? AND expires_at>?", (token, now))
    if not row:
        return None
    path = Path(row["path"])
    if not path.exists() or not path.is_file():
        return None
    if not any(_path_under_root(path, root) for root in MEDIA_ROOTS.values()):
        return None
    row["path_obj"] = path
    row["media_type"] = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return row

