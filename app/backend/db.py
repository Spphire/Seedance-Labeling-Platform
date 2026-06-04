from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from .paths import DB_PATH, ensure_dirs


def _open_connection() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = _open_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def now() -> float:
    return time.time()


def init_db() -> None:
    with connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                uuid TEXT PRIMARY KEY,
                remote_path TEXT NOT NULL,
                local_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                head_video_path TEXT,
                final_video_path TEXT,
                final_status TEXT NOT NULL DEFAULT 'missing',
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS clips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_uuid TEXT NOT NULL REFERENCES episodes(uuid) ON DELETE CASCADE,
                clip_index INTEGER NOT NULL,
                start_sec REAL NOT NULL,
                duration_sec REAL NOT NULL,
                local_path TEXT NOT NULL,
                public_url TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(episode_uuid, clip_index)
            );

            CREATE TABLE IF NOT EXISTS generation_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
                mode TEXT NOT NULL,
                requested_duration_sec INTEGER NOT NULL,
                operator_id TEXT,
                operator_name TEXT,
                prompt TEXT,
                reference_images_json TEXT,
                task_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                output_url TEXT,
                output_path TEXT,
                error TEXT,
                started_at REAL,
                completed_at REAL,
                estimated_total_sec REAL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS seedance_api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER REFERENCES generation_jobs(id) ON DELETE SET NULL,
                clip_id INTEGER REFERENCES clips(id) ON DELETE SET NULL,
                operator_id TEXT,
                operator_name TEXT,
                call_type TEXT NOT NULL DEFAULT 'create_task',
                status TEXT NOT NULL,
                task_id TEXT,
                model TEXT,
                requested_duration_sec INTEGER,
                clip_duration_sec REAL,
                usage_json TEXT,
                raw_response_json TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
                job_id INTEGER REFERENCES generation_jobs(id) ON DELETE SET NULL,
                operator_id TEXT,
                operator_name TEXT,
                decision TEXT NOT NULL,
                note TEXT,
                accepted_path TEXT,
                reviewed_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS resource_locks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                owner_name TEXT NOT NULL,
                token TEXT NOT NULL UNIQUE,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(resource_type, resource_id)
            );

            CREATE INDEX IF NOT EXISTS idx_resource_locks_expires_at
            ON resource_locks(expires_at);

            CREATE INDEX IF NOT EXISTS idx_seedance_api_calls_operator
            ON seedance_api_calls(operator_id, created_at);

            CREATE INDEX IF NOT EXISTS idx_seedance_api_calls_clip
            ON seedance_api_calls(clip_id, created_at);
            """
        )
        _ensure_column(conn, "generation_jobs", "started_at", "REAL")
        _ensure_column(conn, "generation_jobs", "completed_at", "REAL")
        _ensure_column(conn, "generation_jobs", "estimated_total_sec", "REAL")
        _ensure_column(conn, "generation_jobs", "operator_id", "TEXT")
        _ensure_column(conn, "generation_jobs", "operator_name", "TEXT")
        _ensure_column(conn, "generation_jobs", "prompt", "TEXT")
        _ensure_column(conn, "generation_jobs", "reference_images_json", "TEXT")
        _ensure_column(conn, "reviews", "operator_id", "TEXT")
        _ensure_column(conn, "reviews", "operator_name", "TEXT")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reviews_operator
            ON reviews(operator_id, reviewed_at)
            """
        )


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def rows(sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    with connect() as conn:
        return [dict(row) for row in conn.execute(sql, tuple(params)).fetchall()]


def one(sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(sql, tuple(params)).fetchone()
        return dict(row) if row else None
