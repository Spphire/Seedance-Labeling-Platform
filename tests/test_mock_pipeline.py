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
from app.backend.main import app
from app.backend.nedf import fetch_episode
from app.backend.paths import ACCEPTED_DIR, CLIPS_DIR, DATA_DIR, DB_PATH, EPISODES_DIR, FINAL_DIR, GENERATED_DIR, HEAD_VIDEOS_DIR, REFERENCE_IMAGES_DIR
from app.backend.services import create_clips, list_clips, queue_generation, refresh_clip_public_urls, review_clip, run_generation
from app.backend.settings import DEFAULT_SETTINGS, save_settings
from app.backend.video import ffmpeg_probe_fallback, ffprobe_json, run_ffmpeg


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
            path.write_bytes(b"fake-png")
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
        self.assertEqual(payload["content"][-1]["type"], "video_url")

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

    def test_public_settings_do_not_expose_api_key(self) -> None:
        save_settings({"seedance_api_key": "secret-token"})
        res = self.client.get("/api/settings")
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertNotIn("seedance_api_key", data)
        self.assertTrue(data["seedance_api_key_set"])

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
