from __future__ import annotations

import secrets
from typing import Any

from . import db


DEFAULT_LOCK_TTL_SEC = 90
MAX_LOCK_TTL_SEC = 300


class LockError(Exception):
    def __init__(self, status_code: int, message: str, lock: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self.message = message
        self.lock = lock
        super().__init__(message)

    def detail(self) -> dict[str, Any]:
        return {"message": self.message, "lock": self.lock}


def normalize_resource(resource_type: str, resource_id: str | int) -> tuple[str, str]:
    resource_type = str(resource_type).strip().lower()
    if resource_type not in {"clip", "episode"}:
        raise ValueError("resource_type must be clip or episode")
    resource_id_text = str(resource_id).strip().lower()
    if not resource_id_text:
        raise ValueError("resource_id is required")
    return resource_type, resource_id_text


def normalize_ttl(ttl_sec: int | None) -> int:
    if ttl_sec is None:
        return DEFAULT_LOCK_TTL_SEC
    return max(15, min(MAX_LOCK_TTL_SEC, int(ttl_sec)))


def cleanup_expired(conn: Any, now: float | None = None) -> None:
    conn.execute("DELETE FROM resource_locks WHERE expires_at <= ?", (now or db.now(),))


def public_lock(row: dict[str, Any] | Any | None, now: float | None = None, include_token: bool = False) -> dict[str, Any] | None:
    if row is None:
        return None
    if not isinstance(row, dict):
        row = dict(row)
    current = now or db.now()
    data = {
        "resource_type": row["resource_type"],
        "resource_id": row["resource_id"],
        "owner_id": row["owner_id"],
        "owner_name": row["owner_name"],
        "expires_at": row["expires_at"],
        "expires_in_sec": max(0.0, float(row["expires_at"]) - current),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    if include_token:
        data["token"] = row["token"]
    return data


def list_locks() -> list[dict[str, Any]]:
    now = db.now()
    with db.connect() as conn:
        cleanup_expired(conn, now)
        rows = conn.execute(
            "SELECT * FROM resource_locks WHERE expires_at > ? ORDER BY resource_type, resource_id",
            (now,),
        ).fetchall()
    return [public_lock(row, now) for row in rows if row is not None]


def active_lock(resource_type: str, resource_id: str | int) -> dict[str, Any] | None:
    resource_type, resource_id = normalize_resource(resource_type, resource_id)
    now = db.now()
    with db.connect() as conn:
        cleanup_expired(conn, now)
        row = conn.execute(
            "SELECT * FROM resource_locks WHERE resource_type=? AND resource_id=? AND expires_at > ?",
            (resource_type, resource_id, now),
        ).fetchone()
    return public_lock(row, now)


def locks_by_resource(resource_type: str) -> dict[str, dict[str, Any]]:
    resource_type, _ = normalize_resource(resource_type, "placeholder")
    return {lock["resource_id"]: lock for lock in list_locks() if lock["resource_type"] == resource_type}


def acquire_lock(
    resource_type: str,
    resource_id: str | int,
    owner_id: str,
    owner_name: str,
    ttl_sec: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resource_type, resource_id = normalize_resource(resource_type, resource_id)
    owner_id = owner_id.strip()
    owner_name = owner_name.strip() or owner_id
    if not owner_id:
        raise ValueError("owner_id is required")
    ttl = normalize_ttl(ttl_sec)
    now = db.now()
    expires_at = now + ttl
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cleanup_expired(conn, now)
        existing = conn.execute(
            "SELECT * FROM resource_locks WHERE resource_type=? AND resource_id=?",
            (resource_type, resource_id),
        ).fetchone()
        if existing and existing["owner_id"] == owner_id:
            conn.execute(
                """
                UPDATE resource_locks
                SET owner_name=?, expires_at=?, updated_at=?
                WHERE id=?
                """,
                (owner_name, expires_at, now, existing["id"]),
            )
            row = conn.execute("SELECT * FROM resource_locks WHERE id=?", (existing["id"],)).fetchone()
            return public_lock(row, now, include_token=True) | {"acquired": True, "reused": True}
        if existing and not force:
            raise LockError(409, "resource is locked by another reviewer", public_lock(existing, now))
        if existing:
            conn.execute("DELETE FROM resource_locks WHERE id=?", (existing["id"],))
        token = secrets.token_urlsafe(24)
        conn.execute(
            """
            INSERT INTO resource_locks(resource_type, resource_id, owner_id, owner_name, token, expires_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (resource_type, resource_id, owner_id, owner_name, token, expires_at, now, now),
        )
        row = conn.execute(
            "SELECT * FROM resource_locks WHERE resource_type=? AND resource_id=?",
            (resource_type, resource_id),
        ).fetchone()
    return public_lock(row, now, include_token=True) | {"acquired": True, "reused": False}


def renew_lock(token: str, owner_id: str, ttl_sec: int | None = None) -> dict[str, Any]:
    if not token:
        raise LockError(423, "lock token is required")
    owner_id = owner_id.strip()
    ttl = normalize_ttl(ttl_sec)
    now = db.now()
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cleanup_expired(conn, now)
        row = conn.execute("SELECT * FROM resource_locks WHERE token=?", (token,)).fetchone()
        if not row:
            raise LockError(423, "lock expired or was released")
        if row["owner_id"] != owner_id:
            raise LockError(409, "lock belongs to another reviewer", public_lock(row, now))
        conn.execute(
            "UPDATE resource_locks SET expires_at=?, updated_at=? WHERE token=?",
            (now + ttl, now, token),
        )
        updated = conn.execute("SELECT * FROM resource_locks WHERE token=?", (token,)).fetchone()
    return public_lock(updated, now, include_token=True) | {"renewed": True}


def release_lock(token: str, owner_id: str) -> dict[str, Any]:
    if not token:
        return {"released": False}
    now = db.now()
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cleanup_expired(conn, now)
        row = conn.execute("SELECT * FROM resource_locks WHERE token=?", (token,)).fetchone()
        if not row:
            return {"released": False}
        if row["owner_id"] != owner_id:
            raise LockError(409, "lock belongs to another reviewer", public_lock(row, now))
        conn.execute("DELETE FROM resource_locks WHERE token=?", (token,))
    return {"released": True}


def require_lock(resource_type: str, resource_id: str | int, token: str | None) -> dict[str, Any]:
    resource_type, resource_id = normalize_resource(resource_type, resource_id)
    if not token:
        lock = active_lock(resource_type, resource_id)
        raise LockError(423, "lock token is required", lock)
    now = db.now()
    with db.connect() as conn:
        cleanup_expired(conn, now)
        row = conn.execute(
            "SELECT * FROM resource_locks WHERE resource_type=? AND resource_id=? AND expires_at > ?",
            (resource_type, resource_id, now),
        ).fetchone()
    if not row:
        raise LockError(423, "lock expired or was released")
    if row["token"] != token:
        raise LockError(409, "resource is locked by another reviewer", public_lock(row, now))
    return public_lock(row, now)


def require_no_active_lock(resource_type: str, resource_id: str | int) -> None:
    lock = active_lock(resource_type, resource_id)
    if lock:
        raise LockError(409, "resource is locked by another reviewer", lock)
