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
                final_dataset_path TEXT,
                final_dataset_status TEXT NOT NULL DEFAULT 'missing',
                final_dataset_error TEXT,
                preview_video_path TEXT,
                preview_status TEXT NOT NULL DEFAULT 'missing',
                preview_version INTEGER NOT NULL DEFAULT 0,
                preview_error TEXT,
                continuity_state TEXT NOT NULL DEFAULT 'select_anchor',
                anchor_clip_id INTEGER,
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
                source_start_sec REAL,
                source_duration_sec REAL,
                overlap_sec REAL NOT NULL DEFAULT 0,
                timeline_duration_sec REAL,
                timeline_start_sec REAL,
                timeline_end_sec REAL,
                input_timeline_start_sec REAL,
                input_timeline_end_sec REAL,
                direction TEXT NOT NULL DEFAULT 'forward',
                input_kind TEXT NOT NULL DEFAULT 'split',
                anchor_stage TEXT NOT NULL DEFAULT '',
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
                api_key_id TEXT,
                api_key_name TEXT,
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

            CREATE TABLE IF NOT EXISTS media_tokens (
                token TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                purpose TEXT NOT NULL,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lab_experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                prompt TEXT NOT NULL,
                reference_images_json TEXT NOT NULL DEFAULT '[]',
                source_video_path TEXT,
                source_duration_sec REAL,
                input_video_path TEXT,
                clip_start_sec REAL NOT NULL DEFAULT 0,
                clip_duration_sec REAL NOT NULL DEFAULT 4,
                status TEXT NOT NULL DEFAULT 'draft',
                mode TEXT NOT NULL DEFAULT 'mock',
                latest_job_id INTEGER,
                error TEXT,
                operator_id TEXT,
                operator_name TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lab_generation_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL REFERENCES lab_experiments(id) ON DELETE CASCADE,
                mode TEXT NOT NULL,
                requested_duration_sec INTEGER NOT NULL,
                operator_id TEXT,
                operator_name TEXT,
                prompt TEXT,
                reference_images_json TEXT,
                task_id TEXT,
                status TEXT NOT NULL DEFAULT 'queued',
                output_url TEXT,
                output_path TEXT,
                error TEXT,
                started_at REAL,
                completed_at REAL,
                estimated_total_sec REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_resource_locks_expires_at
            ON resource_locks(expires_at);

            CREATE INDEX IF NOT EXISTS idx_media_tokens_expires_at
            ON media_tokens(expires_at);

            CREATE INDEX IF NOT EXISTS idx_seedance_api_calls_operator
            ON seedance_api_calls(operator_id, created_at);

            CREATE INDEX IF NOT EXISTS idx_seedance_api_calls_clip
            ON seedance_api_calls(clip_id, created_at);

            CREATE INDEX IF NOT EXISTS idx_lab_experiments_updated
            ON lab_experiments(updated_at);

            CREATE INDEX IF NOT EXISTS idx_lab_generation_jobs_experiment
            ON lab_generation_jobs(experiment_id, created_at);
            """
        )
        _ensure_column(conn, "generation_jobs", "started_at", "REAL")
        _ensure_column(conn, "generation_jobs", "completed_at", "REAL")
        _ensure_column(conn, "generation_jobs", "estimated_total_sec", "REAL")
        _ensure_column(conn, "generation_jobs", "operator_id", "TEXT")
        _ensure_column(conn, "generation_jobs", "operator_name", "TEXT")
        _ensure_column(conn, "generation_jobs", "prompt", "TEXT")
        _ensure_column(conn, "generation_jobs", "reference_images_json", "TEXT")
        _ensure_column(conn, "seedance_api_calls", "api_key_id", "TEXT")
        _ensure_column(conn, "seedance_api_calls", "api_key_name", "TEXT")
        _ensure_column(conn, "seedance_api_calls", "lab_job_id", "INTEGER")
        _ensure_column(conn, "clips", "source_start_sec", "REAL")
        _ensure_column(conn, "clips", "source_duration_sec", "REAL")
        _ensure_column(conn, "clips", "overlap_sec", "REAL NOT NULL DEFAULT 0")
        _ensure_column(conn, "clips", "timeline_duration_sec", "REAL")
        _ensure_column(conn, "clips", "timeline_start_sec", "REAL")
        _ensure_column(conn, "clips", "timeline_end_sec", "REAL")
        _ensure_column(conn, "clips", "input_timeline_start_sec", "REAL")
        _ensure_column(conn, "clips", "input_timeline_end_sec", "REAL")
        _ensure_column(conn, "clips", "direction", "TEXT NOT NULL DEFAULT 'forward'")
        _ensure_column(conn, "clips", "input_kind", "TEXT NOT NULL DEFAULT 'split'")
        _ensure_column(conn, "clips", "anchor_stage", "TEXT NOT NULL DEFAULT ''")
        conn.execute("UPDATE clips SET source_start_sec=start_sec WHERE source_start_sec IS NULL")
        conn.execute("UPDATE clips SET source_duration_sec=duration_sec WHERE source_duration_sec IS NULL")
        conn.execute("UPDATE clips SET timeline_duration_sec=duration_sec WHERE timeline_duration_sec IS NULL")
        conn.execute("UPDATE clips SET timeline_start_sec=source_start_sec WHERE timeline_start_sec IS NULL")
        conn.execute(
            "UPDATE clips SET timeline_end_sec=timeline_start_sec + timeline_duration_sec WHERE timeline_end_sec IS NULL"
        )
        conn.execute("UPDATE clips SET input_timeline_start_sec=start_sec WHERE input_timeline_start_sec IS NULL")
        conn.execute(
            "UPDATE clips SET input_timeline_end_sec=input_timeline_start_sec + duration_sec WHERE input_timeline_end_sec IS NULL"
        )
        conn.execute("UPDATE clips SET direction='forward' WHERE direction IS NULL OR direction=''")
        conn.execute("UPDATE clips SET input_kind='split' WHERE input_kind IS NULL OR input_kind=''")
        conn.execute(
            """
            UPDATE clips
            SET anchor_stage='official'
            WHERE input_kind='anchor'
              AND (anchor_stage IS NULL OR anchor_stage='')
              AND id IN (
                SELECT anchor_clip_id FROM episodes
                WHERE anchor_clip_id IS NOT NULL
              )
            """
        )
        conn.execute(
            """
            UPDATE clips
            SET anchor_stage='replace_arm'
            WHERE input_kind='anchor' AND (anchor_stage IS NULL OR anchor_stage='')
            """
        )
        _ensure_column(conn, "episodes", "preview_video_path", "TEXT")
        _ensure_column(conn, "episodes", "final_dataset_path", "TEXT")
        _ensure_column(conn, "episodes", "final_dataset_status", "TEXT NOT NULL DEFAULT 'missing'")
        _ensure_column(conn, "episodes", "final_dataset_error", "TEXT")
        _ensure_column(conn, "episodes", "preview_status", "TEXT NOT NULL DEFAULT 'missing'")
        _ensure_column(conn, "episodes", "preview_version", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "episodes", "preview_error", "TEXT")
        _ensure_column(conn, "episodes", "continuity_state", "TEXT NOT NULL DEFAULT 'select_anchor'")
        _ensure_column(conn, "episodes", "anchor_clip_id", "INTEGER")
        conn.execute("UPDATE episodes SET final_dataset_status='missing' WHERE final_dataset_status IS NULL OR final_dataset_status=''")
        conn.execute("UPDATE episodes SET preview_status='missing' WHERE preview_status IS NULL OR preview_status=''")
        conn.execute("UPDATE episodes SET preview_version=0 WHERE preview_version IS NULL")
        conn.execute(
            """
            UPDATE episodes
            SET continuity_state='select_anchor'
            WHERE continuity_state IS NULL OR continuity_state=''
            """
        )
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
