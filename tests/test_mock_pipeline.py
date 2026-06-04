from __future__ import annotations

import json
import os
import shutil
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["SEEDANCE_PLATFORM_ROOT"] = str(Path(__file__).resolve().parent / "_tmp_platform")

from fastapi.testclient import TestClient

from app.backend import db
from app.backend.main import FRONTEND_DIR, app
from app.backend.nedf import fetch_episode
from app.backend.paths import ACCEPTED_DIR, CLIPS_DIR, DATA_DIR, DB_PATH, EPISODES_DIR, FINAL_DIR, GENERATED_DIR, HEAD_VIDEOS_DIR, REFERENCE_IMAGES_DIR
from app.backend.services import (
    create_clips,
    list_episodes,
    list_clips,
    preprocess_one,
    queue_generation,
    queue_stitch_episode,
    refresh_clip_public_urls,
    review_clip,
    run_generation,
)
from app.backend.settings import DEFAULT_PROMPT, DEFAULT_SETTINGS, SETTINGS_PATH, load_settings, public_settings, save_settings
from app.backend.video import ffmpeg_probe_fallback, ffprobe_json, run_ffmpeg
from app.seedance.client import SeedanceClient, resolve_image_value


class MockPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        save_settings(dict(DEFAULT_SETTINGS))
        self.unlink_with_retry(DB_PATH)
        for directory in [CLIPS_DIR, GENERATED_DIR, HEAD_VIDEOS_DIR, ACCEPTED_DIR, FINAL_DIR]:
            self.rmtree_with_retry(directory)
            directory.mkdir(parents=True, exist_ok=True)
        db.init_db()
        self.client = TestClient(app)

    def unlink_with_retry(self, path: Path) -> None:
        for attempt in range(20):
            try:
                path.unlink(missing_ok=True)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.1)

    def rmtree_with_retry(self, path: Path) -> None:
        if not path.exists():
            return
        for attempt in range(20):
            try:
                shutil.rmtree(path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.1)

    def wait_for_final_ready(self, uuid: str, timeout_sec: float = 10.0) -> dict:
        deadline = time.time() + timeout_sec
        last = None
        while time.time() < deadline:
            last = db.one("SELECT * FROM episodes WHERE uuid=?", (uuid,))
            if last and last["final_status"] == "ready":
                return last
            time.sleep(0.1)
        self.fail(f"final did not become ready, last episode state: {last}")

    def make_video(self, path: Path, duration: float) -> None:
        run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size=760x570:rate=24:duration={duration}",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ]
        )

    def make_reference_images(self) -> list[str]:
        REFERENCE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        result = []
        for index in range(4):
            path = REFERENCE_IMAGES_DIR / f"ref-{index + 1}.png"
            path.write_bytes(f"fake-png-{index + 1}".encode("ascii"))
            result.append(path)
        return [str(path) for path in result]

    def make_episode_with_generated_clip(self, uuid: str, duration: float = 5.0) -> tuple[list[dict], list[dict]]:
        now = db.now()
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, duration)
        clips = create_clips(uuid, head, duration)
        results = run_generation(mode="mock")
        return clips, results

    def test_mock_generation_accept_and_stitch(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000001"
        now = db.now()
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, 8)
        clips = create_clips(uuid, head, 8)
        self.assertEqual(len(clips), 1)

        results = run_generation(mode="mock")
        self.assertEqual(results[0]["status"], "succeeded")
        jobs = db.rows("SELECT * FROM generation_jobs")
        self.assertEqual(jobs[0]["mode"], "mock")
        self.assertTrue(Path(jobs[0]["output_path"]).exists())

        res = self.client.post("/api/test/auto_accept", json={"uuid": uuid})
        self.assertEqual(res.status_code, 200, res.text)
        episode = db.one("SELECT * FROM episodes WHERE uuid=?", (uuid,))
        self.assertEqual(episode["final_status"], "stitching")
        episode = self.wait_for_final_ready(uuid)
        final = Path(episode["final_video_path"])
        self.assertTrue(final.exists())
        info = ffprobe_json(final) or ffmpeg_probe_fallback(final)
        stream = info["streams"][0]
        self.assertEqual(stream["codec_name"], "h264")
        self.assertEqual(stream["width"], 760)
        self.assertEqual(stream["height"], 570)
        self.assertEqual(stream["avg_frame_rate"], "30/1")
        self.assertAlmostEqual(float(info["format"]["duration"]), 8.0, delta=0.25)

    def test_frontend_label_and_admin_routes_are_served(self) -> None:
        FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
        (FRONTEND_DIR / "index.html").write_text("<html>runPrompt 管理员全局视图</html>", encoding="utf-8")
        label = self.client.get("/label")
        admin = self.client.get("/admin")

        self.assertEqual(label.status_code, 200, label.text)
        self.assertEqual(admin.status_code, 200, admin.text)
        self.assertIn("runPrompt", label.text)
        self.assertIn("管理员全局视图", admin.text)

    def test_seedance_dry_run_writes_payload_only(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000002"
        now = db.now()
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, 5.2)
        create_clips(uuid, head, 5.2)
        save_settings({"reference_images": ["data:image/png;base64,AA=="]})
        result = run_generation(mode="seedance", dry_run=True)
        payload_path = Path(result[0]["output_path"])
        self.assertEqual(payload_path.suffix, ".json")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["duration"], 6)
        self.assertIn("video_url", json.dumps(payload))

    def test_preprocessed_episode_with_missing_files_is_marked_damaged(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000018"
        now = db.now()
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, head_video_path, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?, ?)
                """,
                (uuid, "mock", "mock", str(HEAD_VIDEOS_DIR / "missing.mp4"), now, now),
            )

        episode = next(item for item in list_episodes() if item["uuid"] == uuid)

        self.assertEqual(episode["preprocess_health"], "damaged")
        self.assertIn("head video file missing", episode["preprocess_health_reason"])
        self.assertEqual(episode["episode_stage"], "预处理文件疑似损坏，需要重新预处理")

    def test_seedance_dry_run_includes_reference_images_in_order(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000012"
        now = db.now()
        refs = self.make_reference_images()
        save_settings({"reference_images": [refs[2], refs[0], refs[3], refs[1]]})
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, 5.2)
        create_clips(uuid, head, 5.2)
        result = run_generation(mode="seedance", dry_run=True)
        payload = json.loads(Path(result[0]["output_path"]).read_text(encoding="utf-8"))

        image_items = [item for item in payload["content"] if item["type"] == "image_url"]
        self.assertEqual(len(image_items), 4)
        self.assertTrue(payload["content"][0]["text"].startswith("把@视频1"))
        self.assertEqual(
            [item["image_url"]["url"] for item in image_items],
            [resolve_image_value(item) for item in [refs[2], refs[0], refs[3], refs[1]]],
        )
        self.assertEqual(payload["content"][-1]["type"], "video_url")

    def test_generation_api_dry_run_uses_prompt_and_reference_overrides(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000020"
        now = db.now()
        refs = self.make_reference_images()
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, 5.2)
        clips = create_clips(uuid, head, 5.2)

        res = self.client.post(
            "/api/generation/run",
            json={
                "mode": "seedance",
                "dry_run": True,
                "prompt": "custom prompt",
                "reference_images": [refs[1], refs[3]],
                "operator_id": "client-a",
                "operator_name": "Alice",
            },
        )

        self.assertEqual(res.status_code, 200, res.text)
        payload = json.loads(Path(res.json()[0]["output_path"]).read_text(encoding="utf-8"))
        image_items = [item for item in payload["content"] if item["type"] == "image_url"]
        self.assertEqual(payload["content"][0]["text"], "custom prompt")
        self.assertEqual([item["image_url"]["url"] for item in image_items], [resolve_image_value(refs[1]), resolve_image_value(refs[3])])
        job = db.one("SELECT * FROM generation_jobs WHERE clip_id=?", (clips[0]["id"],))
        self.assertEqual(job["prompt"], "custom prompt")
        self.assertEqual(json.loads(job["reference_images_json"]), [refs[1], refs[3]])

    def test_default_reference_images_are_seedance_prompt_order(self) -> None:
        settings = load_settings()
        self.assertEqual(
            settings["reference_images"],
            [
                "app/reference_images/l-near-iphone.png",
                "app/reference_images/r-near-iphone.png",
                "app/reference_images/l-far-iphone.png",
                "app/reference_images/r-far-iphone.png",
            ],
        )
        self.assertEqual(settings["default_generation_preset_id"], "iphone-default")
        self.assertEqual(settings["generation_presets"][0]["id"], "iphone-default")
        self.assertEqual(settings["generation_presets"][0]["prompt"], DEFAULT_PROMPT)
        self.assertEqual(settings["generation_presets"][0]["reference_images"], settings["reference_images"])

    def test_old_reference_image_names_are_migrated_to_iphone_names(self) -> None:
        save_settings(
            {
                "reference_images": [
                    "app/reference_images/l-near.png",
                    "app/reference_images/r-near.png",
                    "app/reference_images/l-far.png",
                    "app/reference_images/r-far.png",
                ]
            }
        )

        settings = load_settings()
        persisted = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))

        self.assertEqual(settings["reference_images"], DEFAULT_SETTINGS["reference_images"])
        self.assertEqual(persisted["reference_images"], DEFAULT_SETTINGS["reference_images"])

    def test_reference_image_names_inside_presets_are_migrated(self) -> None:
        save_settings(
            {
                "default_generation_preset_id": "old-iphone",
                "generation_presets": [
                    {
                        "id": "old-iphone",
                        "name": "旧 iPhone 组合",
                        "prompt": DEFAULT_PROMPT,
                        "reference_images": [
                            "app/reference_images/l-near.png",
                            "app/reference_images/r-near.png",
                            "app/reference_images/l-far.png",
                            "app/reference_images/r-far.png",
                        ],
                    }
                ],
            }
        )

        settings = load_settings()
        persisted = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))

        self.assertEqual(settings["default_generation_preset_id"], "old-iphone")
        self.assertEqual(settings["generation_presets"][0]["reference_images"], DEFAULT_SETTINGS["reference_images"])
        self.assertEqual(persisted["generation_presets"][0]["reference_images"], DEFAULT_SETTINGS["reference_images"])

    def test_seedance_queue_marks_running_and_blocks_duplicate(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000011"
        now = db.now()
        save_settings({"generation_mode": "seedance", "seedance_seconds_per_video_second": 24})
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, 5.2)
        clips = create_clips(uuid, head, 5.2)

        with patch("app.backend.services._GENERATION_EXECUTOR.submit") as submit:
            first = queue_generation(mode="seedance")
            second = queue_generation(mode="seedance")

        self.assertEqual(first[0]["status"], "queued")
        self.assertEqual(first[0]["estimated_total_sec"], 144)
        self.assertEqual(second, [])
        submit.assert_called_once()

        clip = db.one("SELECT * FROM clips WHERE id=?", (clips[0]["id"],))
        self.assertEqual(clip["status"], "generating")
        job = db.one("SELECT * FROM generation_jobs WHERE clip_id=?", (clips[0]["id"],))
        self.assertEqual(job["status"], "running")
        self.assertEqual(job["estimated_total_sec"], 144)
        listed = [item for item in list_clips() if item["id"] == clips[0]["id"]][0]
        self.assertEqual(listed["latest_job"]["status"], "running")
        self.assertIsNone(listed["generated_url"])

    def test_bulk_generation_includes_rejected_clips(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000016"
        now = db.now()
        statuses = ["pending", "generated_failed", "rejected", "generated", "accepted", "flagged", "generating"]
        clip_ids_by_index = {}
        save_settings({"mock_async": True, "mock_seconds_per_video_second": 0.02})
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
            for index, status in enumerate(statuses):
                cur = conn.execute(
                    """
                    INSERT INTO clips(
                        episode_uuid, clip_index, start_sec, duration_sec,
                        local_path, public_url, status, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid,
                        index,
                        float(index),
                        5.0,
                        str((CLIPS_DIR / uuid / f"clip_{index:04d}.mp4").resolve()),
                        f"http://localhost/clips/{uuid}/clip_{index:04d}.mp4",
                        status,
                        now,
                        now,
                    ),
                )
                clip_ids_by_index[index] = cur.lastrowid

        with patch("app.backend.services._GENERATION_EXECUTOR.submit") as submit:
            queued = queue_generation(mode="mock")

        queued_clip_ids = {item["clip_id"] for item in queued}
        expected_clip_ids = {clip_ids_by_index[0], clip_ids_by_index[1], clip_ids_by_index[2]}
        self.assertEqual(len(queued), 3)
        self.assertEqual(queued_clip_ids, expected_clip_ids)
        self.assertEqual(submit.call_count, 3)

        final_statuses = {
            row["clip_index"]: row["status"]
            for row in db.rows("SELECT clip_index, status FROM clips WHERE episode_uuid=?", (uuid,))
        }
        self.assertEqual(final_statuses[0], "generating")
        self.assertEqual(final_statuses[1], "generating")
        self.assertEqual(final_statuses[2], "generating")
        self.assertEqual(final_statuses[3], "generated")
        self.assertEqual(final_statuses[4], "accepted")
        self.assertEqual(final_statuses[5], "flagged")
        self.assertEqual(final_statuses[6], "generating")

    def test_retry_api_uses_explicit_mode_instead_of_saved_default(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000014"
        now = db.now()
        save_settings({"generation_mode": "seedance", "mock_async": False})
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, 5.0)
        clips = create_clips(uuid, head, 5.0)
        clip_id = clips[0]["id"]
        lock = self.client.post(
            "/api/locks/acquire",
            json={"resource_type": "clip", "resource_id": str(clip_id), "owner_id": "alice", "owner_name": "Alice"},
        )
        self.assertEqual(lock.status_code, 200, lock.text)
        with db.connect() as conn:
            conn.execute("UPDATE clips SET status='generated_failed' WHERE id=?", (clip_id,))

        res = self.client.post(
            f"/api/clips/{clip_id}/retry",
            json={"lock_token": lock.json()["token"], "mode": "mock"},
        )

        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()["status"], "succeeded")
        job = db.one("SELECT * FROM generation_jobs WHERE id=?", (res.json()["job_id"],))
        self.assertEqual(job["mode"], "mock")

    def test_generation_api_defaults_to_mock_even_when_saved_mode_is_seedance(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000017"
        now = db.now()
        save_settings({"generation_mode": "seedance", "mock_async": True})
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, 5.0)
        clips = create_clips(uuid, head, 5.0)

        with patch("app.backend.services._GENERATION_EXECUTOR.submit") as submit:
            res = self.client.post(
                "/api/generation/run",
                json={"operator_id": "client-a", "operator_name": "Alice"},
            )

        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()[0]["status"], "queued")
        submit.assert_called_once()
        job = db.one("SELECT * FROM generation_jobs WHERE clip_id=?", (clips[0]["id"],))
        self.assertEqual(job["mode"], "mock")
        self.assertEqual(job["operator_id"], "client-a")
        self.assertEqual(job["operator_name"], "Alice")

    def test_seedance_usage_endpoint_summarizes_calls(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000019"
        now = db.now()
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
            cur = conn.execute(
                """
                INSERT INTO clips(episode_uuid, clip_index, start_sec, duration_sec, local_path, public_url, status, created_at, updated_at)
                VALUES (?, 0, 0, 5, ?, ?, 'generated', ?, ?)
                """,
                (uuid, str(CLIPS_DIR / uuid / "clip_0000.mp4"), "http://localhost/clip.mp4", now, now),
            )
            clip_id = cur.lastrowid
            job_cur = conn.execute(
                """
                INSERT INTO generation_jobs(
                    clip_id, mode, requested_duration_sec, operator_id, operator_name,
                    task_id, status, started_at, completed_at, created_at, updated_at
                )
                VALUES (?, 'seedance', 6, 'client-a', 'Alice', 'task-1', 'succeeded', ?, ?, ?, ?)
                """,
                (clip_id, now, now, now, now),
            )
            conn.execute(
                """
                INSERT INTO seedance_api_calls(
                    job_id, clip_id, operator_id, operator_name, status, task_id,
                    model, requested_duration_sec, clip_duration_sec, usage_json,
                    raw_response_json, created_at, updated_at
                )
                VALUES (?, ?, 'client-a', 'Alice', 'succeeded', 'task-1', 'seedance-fast', 6, 5, ?, ?, ?, ?)
                """,
                (job_cur.lastrowid, clip_id, json.dumps({"total_tokens": 123}), json.dumps({"large": True}), now, now),
            )

        res = self.client.get("/api/usage/seedance")

        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["summary"][0]["operator_id"], "client-a")
        self.assertEqual(data["summary"][0]["call_count"], 1)
        self.assertEqual(data["summary"][0]["requested_duration_sec"], 6)
        self.assertEqual(data["recent_calls"][0]["usage"], {"total_tokens": 123})
        self.assertNotIn("raw_response_json", data["recent_calls"][0])

    def test_reviewer_activity_endpoint_summarizes_review_operator(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000021"
        clips, results = self.make_episode_with_generated_clip(uuid, 5.0)
        clip_id = clips[0]["id"]
        lock = self.client.post(
            "/api/locks/acquire",
            json={"resource_type": "clip", "resource_id": str(clip_id), "owner_id": "client-a", "owner_name": "Alice"},
        )
        self.assertEqual(lock.status_code, 200, lock.text)

        review = self.client.post(
            f"/api/review/{clip_id}",
            json={
                "decision": "accept",
                "job_id": results[0]["job_id"],
                "lock_token": lock.json()["token"],
                "operator_id": "client-a",
                "operator_name": "Alice",
            },
        )
        self.assertEqual(review.status_code, 200, review.text)

        res = self.client.get("/api/reviews/activity")

        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["summary"][0]["operator_id"], "client-a")
        self.assertEqual(data["summary"][0]["review_count"], 1)
        self.assertEqual(data["summary"][0]["accept_count"], 1)
        self.assertEqual(data["recent_reviews"][0]["operator_name"], "Alice")
        self.assertEqual(data["recent_reviews"][0]["decision"], "accept")

    def test_retry_api_mock_mode_uses_async_timing_when_enabled(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000015"
        now = db.now()
        save_settings({"generation_mode": "seedance", "mock_async": True, "mock_seconds_per_video_second": 0.02})
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, 5.0)
        clips = create_clips(uuid, head, 5.0)
        clip_id = clips[0]["id"]
        lock = self.client.post(
            "/api/locks/acquire",
            json={"resource_type": "clip", "resource_id": str(clip_id), "owner_id": "alice", "owner_name": "Alice"},
        )
        self.assertEqual(lock.status_code, 200, lock.text)
        with db.connect() as conn:
            conn.execute("UPDATE clips SET status='generated_failed' WHERE id=?", (clip_id,))

        res = self.client.post(
            f"/api/clips/{clip_id}/retry",
            json={"lock_token": lock.json()["token"], "mode": "mock"},
        )

        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()["status"], "queued")
        running = db.one("SELECT * FROM generation_jobs WHERE id=?", (res.json()["job_id"],))
        self.assertEqual(running["mode"], "mock")
        self.assertEqual(running["status"], "running")
        self.assertEqual(db.one("SELECT status FROM clips WHERE id=?", (clip_id,))["status"], "generating")

        deadline = time.time() + 5
        job = running
        while time.time() < deadline:
            job = db.one("SELECT * FROM generation_jobs WHERE id=?", (res.json()["job_id"],))
            if job["status"] == "succeeded":
                break
            time.sleep(0.05)
        self.assertEqual(job["status"], "succeeded")
        self.assertTrue(Path(job["output_path"]).exists())

    def test_public_settings_do_not_expose_api_key(self) -> None:
        save_settings({"seedance_api_key": "secret-token"})
        res = self.client.get("/api/settings")
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertNotIn("seedance_api_key", data)
        self.assertTrue(data["seedance_api_key_set"])

    def test_env_api_key_sets_public_flag_without_persisting_secret(self) -> None:
        save_settings({"seedance_api_key": ""})
        with patch.dict(os.environ, {"SEEDANCE_API_KEY": "env-secret"}):
            data = public_settings()
        persisted = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))

        self.assertNotIn("seedance_api_key", data)
        self.assertTrue(data["seedance_api_key_set"])
        self.assertEqual(persisted["seedance_api_key"], "")

    def test_broken_prompt_is_repaired_and_persisted(self) -> None:
        broken = "?@??1???????@??1@??2??????@??1???????@??3@??4???????????????????????"
        save_settings({"default_prompt": broken})
        settings = load_settings()
        persisted = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))

        self.assertEqual(settings["default_prompt"], DEFAULT_PROMPT)
        self.assertEqual(persisted["default_prompt"], DEFAULT_PROMPT)

    def test_replacement_char_prompt_is_repaired_and_persisted(self) -> None:
        save_settings({"default_prompt": "\ufffd\ufffd@\ufffd\ufffd1 broken"})
        settings = load_settings()
        persisted = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))

        self.assertEqual(settings["default_prompt"], DEFAULT_PROMPT)
        self.assertEqual(persisted["default_prompt"], DEFAULT_PROMPT)

    def test_legacy_reference_order_is_repaired_and_persisted(self) -> None:
        REFERENCE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        legacy_orders = [
            [
                "app/reference_images/l-far.png",
                "app/reference_images/l-near.png",
                "app/reference_images/r-far.png",
                "app/reference_images/r-near.png",
            ],
            [
                "app/reference_images/l-far.png",
                "app/reference_images/r-far.png",
                "app/reference_images/l-near.png",
                "app/reference_images/r-near.png",
            ],
        ]

        for legacy in legacy_orders:
            save_settings({"reference_images": legacy})
            settings = load_settings()
            persisted = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))

            self.assertEqual(settings["reference_images"], DEFAULT_SETTINGS["reference_images"])
            self.assertEqual(persisted["reference_images"], DEFAULT_SETTINGS["reference_images"])

    def test_seedance_output_url_prefers_task_content_video_url(self) -> None:
        input_url = "http://106.14.2.243:18080/clips/episode/clip_0000.mp4"
        output_url = "https://ark-acg-cn-beijing.tos-cn-beijing.volces.com/result.mp4?signature=ok"
        task = {
            "status": "succeeded",
            "content": {"video_url": output_url, "file_url": None},
            "request": {"content": [{"type": "video_url", "video_url": {"url": input_url}}]},
        }

        self.assertEqual(SeedanceClient._find_output_url(task, {input_url}), output_url)

    def test_mock_async_marks_running_then_completes(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000013"
        now = db.now()
        save_settings({"mock_async": True, "mock_seconds_per_video_second": 0.02})
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, 5.0)
        clips = create_clips(uuid, head, 5.0)

        queued = queue_generation(mode="mock")
        self.assertEqual(queued[0]["status"], "queued")
        running = db.one("SELECT * FROM generation_jobs WHERE id=?", (queued[0]["job_id"],))
        self.assertEqual(running["status"], "running")
        self.assertEqual(db.one("SELECT status FROM clips WHERE id=?", (clips[0]["id"],))["status"], "generating")

        deadline = time.time() + 5
        job = running
        while time.time() < deadline:
            job = db.one("SELECT * FROM generation_jobs WHERE id=?", (queued[0]["job_id"],))
            if job["status"] == "succeeded":
                break
            time.sleep(0.05)
        self.assertEqual(job["status"], "succeeded")
        self.assertTrue(Path(job["output_path"]).exists())
        self.assertEqual(db.one("SELECT status FROM clips WHERE id=?", (clips[0]["id"],))["status"], "generated")

    def test_fetch_episode_uses_local_mount_without_copying(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000010"
        source_root = DATA_DIR / "local_remote_source"
        source_preprocessed = source_root / uuid / "preprocessed"
        source_preprocessed.mkdir(parents=True, exist_ok=True)
        (source_preprocessed / "metadata.json").write_text("{}", encoding="utf-8")

        local_dir = EPISODES_DIR / uuid
        resolved = fetch_episode("unresolvable-host", str(source_root), uuid, local_dir)

        self.assertEqual(resolved, source_root / uuid)
        self.assertFalse(local_dir.exists())

    def test_preprocess_skips_complete_episode_and_recovers_missing_clip_file(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000017"
        now = db.now()
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, 16)
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, head_video_path, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?, ?)
                """,
                (uuid, "mock", str((EPISODES_DIR / uuid).resolve()), str(head.resolve()), now, now),
            )
        clips = create_clips(uuid, head, 16)
        before_ids = [clip["id"] for clip in clips]
        lock = self.client.post(
            "/api/locks/acquire",
            json={"resource_type": "episode", "resource_id": uuid, "owner_id": "alice", "owner_name": "Alice"},
        )
        self.assertEqual(lock.status_code, 200, lock.text)
        token = lock.json()["token"]

        skipped = preprocess_one(uuid, load_settings(), fetch_remote=False, lock_token=token)

        self.assertEqual(skipped["status"], "skipped")
        self.assertEqual(skipped["clip_count"], len(clips))
        self.assertEqual([row["id"] for row in db.rows("SELECT id FROM clips WHERE episode_uuid=? ORDER BY clip_index", (uuid,))], before_ids)

        Path(clips[0]["path"]).unlink()
        rebuilt = preprocess_one(uuid, load_settings(), fetch_remote=False, lock_token=token)

        self.assertEqual(rebuilt["status"], "preprocessed")
        self.assertIn("clip 0 file is missing", rebuilt["reason"])
        self.assertNotEqual(
            [row["id"] for row in db.rows("SELECT id FROM clips WHERE episode_uuid=? ORDER BY clip_index", (uuid,))],
            before_ids,
        )
        self.assertTrue(Path(rebuilt["clips"][0]["path"]).exists())

    def test_submit_and_preprocess_combined_endpoint_submits_before_preprocess(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000018"
        self.assertIsNone(db.one("SELECT * FROM episodes WHERE uuid=?", (uuid,)))
        lock = self.client.post(
            "/api/locks/acquire",
            json={"resource_type": "episode", "resource_id": uuid, "owner_id": "alice", "owner_name": "Alice"},
        )
        self.assertEqual(lock.status_code, 200, lock.text)
        res = self.client.post(
            "/api/pipeline/submit_preprocess",
            json={"episodes_text": uuid, "fetch_remote": False, "lock_tokens": {uuid: lock.json()["token"]}},
        )
        self.assertEqual(res.status_code, 200, res.text)
        result = res.json()

        self.assertTrue(any(item["uuid"] == uuid for item in result["episodes"]))
        self.assertEqual(result["preprocess"][0]["uuid"], uuid)
        self.assertIn("error", result["preprocess"][0])
        self.assertIsNotNone(db.one("SELECT * FROM episodes WHERE uuid=?", (uuid,)))

    def test_public_base_url_refreshes_existing_clip_records(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000009"
        now = db.now()
        save_settings({"public_base_url": "http://old.example:18080"})
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, 5.2)
        clips = create_clips(uuid, head, 5.2)
        before = db.one("SELECT public_url FROM clips WHERE id=?", (clips[0]["id"],))
        self.assertTrue(before["public_url"].startswith("http://old.example:18080/clips/"))

        save_settings({"public_base_url": "http://106.14.2.243:18080"})
        updated = refresh_clip_public_urls()

        after = db.one("SELECT public_url FROM clips WHERE id=?", (clips[0]["id"],))
        self.assertEqual(updated, 1)
        self.assertTrue(after["public_url"].startswith("http://106.14.2.243:18080/clips/"))

    def test_accept_rejects_dry_run_json_output(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000003"
        now = db.now()
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, 5.2)
        clips = create_clips(uuid, head, 5.2)
        result = run_generation(mode="seedance", dry_run=True)
        with self.assertRaises(ValueError):
            review_clip(clips[0]["id"], "accept", result[0]["job_id"], require_lock_token=False)

    def test_review_job_must_belong_to_clip(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000004"
        now = db.now()
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, 16)
        clips = create_clips(uuid, head, 16)
        result = run_generation(mode="mock", clip_ids=[clips[0]["id"]])
        with self.assertRaises(ValueError):
            review_clip(clips[1]["id"], "accept", result[0]["job_id"], require_lock_token=False)

    def test_stitch_queue_locks_episode_and_clips_until_worker_finishes(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000019"
        now = db.now()
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, 16)
        clips = create_clips(uuid, head, 16)
        with db.connect() as conn:
            conn.execute("UPDATE clips SET status='accepted' WHERE episode_uuid=?", (uuid,))
            conn.execute("UPDATE episodes SET final_status='missing' WHERE uuid=?", (uuid,))

        with patch("app.backend.services._STITCH_EXECUTOR.submit") as submit:
            result = queue_stitch_episode(uuid, check_episode_lock=False)

        self.assertTrue(result["queued"])
        submit.assert_called_once()
        locks = db.rows("SELECT * FROM resource_locks WHERE owner_id='system-stitcher'")
        resources = {(row["resource_type"], row["resource_id"]) for row in locks}
        self.assertIn(("episode", uuid), resources)
        for clip in clips:
            self.assertIn(("clip", str(clip["id"])), resources)

    def test_clip_lock_required_for_review_api(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000005"
        clips, results = self.make_episode_with_generated_clip(uuid)
        clip_id = clips[0]["id"]
        job_id = results[0]["job_id"]

        missing = self.client.post(f"/api/review/{clip_id}", json={"decision": "accept", "job_id": job_id})
        self.assertEqual(missing.status_code, 423, missing.text)

        lock = self.client.post(
            "/api/locks/acquire",
            json={
                "resource_type": "clip",
                "resource_id": str(clip_id),
                "owner_id": "alice",
                "owner_name": "Alice",
            },
        )
        self.assertEqual(lock.status_code, 200, lock.text)
        token = lock.json()["token"]

        conflict = self.client.post(
            f"/api/review/{clip_id}",
            json={"decision": "reject", "job_id": job_id, "lock_token": "wrong-token"},
        )
        self.assertEqual(conflict.status_code, 409, conflict.text)
        self.assertEqual(conflict.json()["detail"]["lock"]["owner_name"], "Alice")

        ok = self.client.post(
            f"/api/review/{clip_id}",
            json={"decision": "flag", "job_id": job_id, "lock_token": token, "note": "needs review"},
        )
        self.assertEqual(ok.status_code, 200, ok.text)
        clip = db.one("SELECT * FROM clips WHERE id=?", (clip_id,))
        self.assertEqual(clip["status"], "flagged")

    def test_lock_renew_release_and_reacquire(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000006"
        clips, _ = self.make_episode_with_generated_clip(uuid)
        clip_id = clips[0]["id"]

        first = self.client.post(
            "/api/locks/acquire",
            json={"resource_type": "clip", "resource_id": str(clip_id), "owner_id": "alice", "owner_name": "Alice"},
        )
        self.assertEqual(first.status_code, 200, first.text)
        token = first.json()["token"]

        blocked = self.client.post(
            "/api/locks/acquire",
            json={"resource_type": "clip", "resource_id": str(clip_id), "owner_id": "bob", "owner_name": "Bob"},
        )
        self.assertEqual(blocked.status_code, 409, blocked.text)

        renewed = self.client.post("/api/locks/renew", json={"token": token, "owner_id": "alice"})
        self.assertEqual(renewed.status_code, 200, renewed.text)
        self.assertEqual(renewed.json()["owner_name"], "Alice")

        released = self.client.post("/api/locks/release", json={"token": token, "owner_id": "alice"})
        self.assertEqual(released.status_code, 200, released.text)
        self.assertTrue(released.json()["released"])

        second = self.client.post(
            "/api/locks/acquire",
            json={"resource_type": "clip", "resource_id": str(clip_id), "owner_id": "bob", "owner_name": "Bob"},
        )
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()["owner_name"], "Bob")

    def test_bulk_generation_skips_locked_clip(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000007"
        now = db.now()
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, 16)
        clips = create_clips(uuid, head, 16)

        locked = self.client.post(
            "/api/locks/acquire",
            json={
                "resource_type": "clip",
                "resource_id": str(clips[0]["id"]),
                "owner_id": "alice",
                "owner_name": "Alice",
            },
        )
        self.assertEqual(locked.status_code, 200, locked.text)

        generated = run_generation(mode="mock")
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0]["clip_id"], clips[1]["id"])
        statuses = {row["id"]: row["status"] for row in db.rows("SELECT id, status FROM clips")}
        self.assertEqual(statuses[clips[0]["id"]], "pending")
        self.assertEqual(statuses[clips[1]["id"]], "generated")

    def test_episode_mutation_blocked_by_active_clip_lock(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000008"
        now = db.now()
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?)
                """,
                (uuid, "mock", "mock", now, now),
            )
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, 5)
        clips = create_clips(uuid, head, 5)
        clip_lock = self.client.post(
            "/api/locks/acquire",
            json={
                "resource_type": "clip",
                "resource_id": str(clips[0]["id"]),
                "owner_id": "alice",
                "owner_name": "Alice",
            },
        )
        self.assertEqual(clip_lock.status_code, 200, clip_lock.text)
        episode_lock = self.client.post(
            "/api/locks/acquire",
            json={"resource_type": "episode", "resource_id": uuid, "owner_id": "bob", "owner_name": "Bob"},
        )
        self.assertEqual(episode_lock.status_code, 200, episode_lock.text)
        res = self.client.post(
            "/api/pipeline/preprocess",
            json={"uuids": [uuid], "fetch_remote": False, "lock_tokens": {uuid: episode_lock.json()["token"]}},
        )
        self.assertEqual(res.status_code, 409, res.text)
        self.assertEqual(res.json()["detail"]["lock"]["owner_name"], "Alice")


if __name__ == "__main__":
    unittest.main()
