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
from app.backend import services as backend_services
from app.backend.main import FRONTEND_DIR, app
from app.backend.nedf import fetch_episode
from app.backend.paths import ACCEPTED_DIR, CLIPS_DIR, DATA_DIR, DB_PATH, EPISODES_DIR, FINAL_DIR, GENERATED_DIR, HEAD_VIDEOS_DIR, REFERENCE_IMAGES_DIR
from app.backend.services import (
    create_anchor_candidates,
    create_clips,
    import_head_video,
    list_episodes,
    list_clips,
    preprocess_one,
    queue_generation,
    queue_preview_episode,
    queue_rolling_generation,
    queue_stitch_episode,
    refresh_clip_public_urls,
    review_clip,
    run_generation,
)
from app.backend.settings import (
    COLLECTOR_ONLY_PRESET_ID,
    COLLECTOR_ONLY_PROMPT,
    COLLECTOR_ONLY_REFERENCE_IMAGES,
    DEFAULT_PROMPT,
    DEFAULT_MOCK_SECONDS_PER_VIDEO_SECOND,
    DEFAULT_SETTINGS,
    DEFAULT_GENERATION_PRESET_ID,
    GENERATION_PRESETS_VERSION,
    IPHONE2DEPLOY_PRESET_ID,
    IPHONE2DEPLOY_PROMPT,
    IPHONE2DEPLOY_REFERENCE_IMAGES,
    SETTINGS_PATH,
    load_settings,
    public_settings,
    save_settings,
)
from app.backend.video import black_video, ffmpeg_probe_fallback, ffprobe_json, run_ffmpeg
from app.seedance.client import SeedanceClient, resolve_image_value


class MockPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.wait_for_background_idle()
        with backend_services._PREVIEW_LOCK, backend_services._STITCH_LOCK:
            backend_services._PREVIEWING_EPISODES.clear()
            backend_services._STITCHING_EPISODES.clear()
        with backend_services._GENERATION_CONDITION:
            backend_services._SEEDANCE_KEY_ACTIVE.clear()
            backend_services._GENERATION_ACTIVE = 0
        save_settings(dict(DEFAULT_SETTINGS))
        self.unlink_with_retry(DB_PATH)
        for directory in [CLIPS_DIR, GENERATED_DIR, HEAD_VIDEOS_DIR, ACCEPTED_DIR, FINAL_DIR]:
            self.rmtree_with_retry(directory)
            directory.mkdir(parents=True, exist_ok=True)
        db.init_db()
        self.client = TestClient(app)

    def wait_for_background_idle(self, timeout_sec: float = 10.0) -> None:
        deadline = time.time() + timeout_sec
        previewing: set[str] = set()
        stitching: set[str] = set()
        while time.time() < deadline:
            with backend_services._PREVIEW_LOCK, backend_services._STITCH_LOCK:
                previewing = set(backend_services._PREVIEWING_EPISODES)
                stitching = set(backend_services._STITCHING_EPISODES)
            if not previewing and not stitching:
                return
            time.sleep(0.1)
        self.fail(f"background workers did not become idle: preview={previewing}, stitch={stitching}")

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

    def wait_for_preview_ready(self, uuid: str, timeout_sec: float = 10.0) -> dict:
        deadline = time.time() + timeout_sec
        last = None
        while time.time() < deadline:
            last = db.one("SELECT * FROM episodes WHERE uuid=?", (uuid,))
            if last and last["preview_status"] == "ready":
                return last
            if last and last["preview_status"] == "failed":
                self.fail(f"preview failed: {last}")
            time.sleep(0.1)
        self.fail(f"preview did not become ready, last episode state: {last}")

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

    def make_reference_files_for(self, refs: list[str]) -> None:
        REFERENCE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        for ref in refs:
            path = REFERENCE_IMAGES_DIR / Path(ref).name
            path.write_bytes(f"fake-{path.name}".encode("ascii"))

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

    def make_head_ready_episode(self, uuid: str, duration: float) -> Path:
        now = db.now()
        head = HEAD_VIDEOS_DIR / f"{uuid}_head_760x570.mp4"
        self.make_video(head, duration)
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes(uuid, remote_path, local_path, status, head_video_path, created_at, updated_at)
                VALUES (?, ?, ?, 'preprocessed', ?, ?, ?)
                """,
                (uuid, "mock", "mock", str(head.resolve()), now, now),
            )
        return head

    def create_anchor_candidates_with_lock(self, uuid: str, starts: list[float]) -> dict:
        lock = self.client.post(
            "/api/locks/acquire",
            json={"resource_type": "episode", "resource_id": uuid, "owner_id": "alice", "owner_name": "Alice"},
        )
        self.assertEqual(lock.status_code, 200, lock.text)
        token = lock.json()["token"]
        try:
            return create_anchor_candidates(uuid, starts, token)
        finally:
            self.client.post("/api/locks/release", json={"token": token, "owner_id": "alice"})

    def accept_single_anchor_candidate_to_official(self, uuid: str) -> tuple[dict, dict]:
        first_result = queue_rolling_generation(mode="mock")
        anchor = db.one("SELECT * FROM clips WHERE episode_uuid=? AND input_kind='anchor'", (uuid,))
        self.assertIsNotNone(anchor)
        first_job = next(item for item in first_result if item.get("clip_id") == anchor["id"])
        review_clip(anchor["id"], "accept", first_job["job_id"], require_lock_token=False)
        second_result = queue_rolling_generation(mode="mock")
        second_job = next(item for item in second_result if item.get("clip_id") == anchor["id"])
        accepted = review_clip(anchor["id"], "accept", second_job["job_id"], require_lock_token=False)
        return db.one("SELECT * FROM clips WHERE id=?", (anchor["id"],)), accepted

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

    def test_import_head_video_prepares_head_without_creating_clips(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000022"
        source = DATA_DIR / "manual_head.mp4"
        self.make_video(source, 8)
        lock = self.client.post(
            "/api/locks/acquire",
            json={"resource_type": "episode", "resource_id": uuid, "owner_id": "alice", "owner_name": "Alice"},
        )
        self.assertEqual(lock.status_code, 200, lock.text)

        result = import_head_video(uuid, str(source), lock.json()["token"])

        self.assertEqual(result["clips"], [])
        self.assertIsNone(db.one("SELECT * FROM clips WHERE episode_uuid=?", (uuid,)))
        episode = db.one("SELECT * FROM episodes WHERE uuid=?", (uuid,))
        self.assertEqual(episode["status"], "preprocessed")
        self.assertEqual(episode["continuity_state"], "select_anchor")
        self.assertIsNone(episode["anchor_clip_id"])
        self.assertTrue(Path(episode["head_video_path"]).exists())

    def test_rolling_generation_runs_anchor_candidates(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000028"
        source = DATA_DIR / "manual_head_locked.mp4"
        save_settings({"mock_async": False})
        self.make_video(source, 8)
        lock = self.client.post(
            "/api/locks/acquire",
            json={"resource_type": "episode", "resource_id": uuid, "owner_id": "alice", "owner_name": "Alice"},
        )
        self.assertEqual(lock.status_code, 200, lock.text)
        import_head_video(uuid, str(source), lock.json()["token"])
        self.client.post("/api/locks/release", json={"token": lock.json()["token"], "owner_id": "alice"})
        created = self.create_anchor_candidates_with_lock(uuid, [0, 4])
        self.assertEqual(len(created["created"]), 2)
        clips = db.rows("SELECT * FROM clips WHERE episode_uuid=? ORDER BY clip_index", (uuid,))
        self.assertEqual([clip["input_kind"] for clip in clips], ["anchor", "anchor"])
        self.assertEqual([clip["status"] for clip in clips], ["pending", "pending"])

        result = queue_rolling_generation(mode="mock")

        succeeded_ids = {item["clip_id"] for item in result if item.get("status") == "succeeded"}
        self.assertEqual(succeeded_ids, {clip["id"] for clip in clips})
        statuses = {row["id"]: row["status"] for row in db.rows("SELECT id, status FROM clips WHERE episode_uuid=?", (uuid,))}
        self.assertEqual(set(statuses.values()), {"generated"})

    def test_anchor_candidate_creation_skips_invalid_starts_without_dropping_valid_ones(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000031"
        self.make_head_ready_episode(uuid, 20)

        result = self.create_anchor_candidates_with_lock(uuid, [1, 4, 8, 8])

        self.assertEqual([clip["timeline_start_sec"] for clip in result["created"]], [4.0, 8.0])
        self.assertEqual(len(result["skipped"]), 2)
        self.assertEqual(result["continuity_state"], "anchor_candidates")
        clips = db.rows("SELECT * FROM clips WHERE episode_uuid=? ORDER BY timeline_start_sec", (uuid,))
        self.assertEqual([clip["input_kind"] for clip in clips], ["anchor", "anchor"])
        self.assertEqual([clip["status"] for clip in clips], ["pending", "pending"])

    def test_rolling_generation_advances_after_accept_and_reject_reruns_same_clip(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000023"
        save_settings({"mock_async": False})
        self.make_head_ready_episode(uuid, 32)
        self.create_anchor_candidates_with_lock(uuid, [14])

        anchor_result = queue_rolling_generation(mode="mock")
        self.assertEqual(anchor_result[0]["status"], "succeeded")
        anchor = db.one("SELECT * FROM clips WHERE episode_uuid=? AND input_kind='anchor'", (uuid,))
        self.assertEqual(anchor["status"], "generated")
        accepted = review_clip(anchor["id"], "accept", anchor_result[0]["job_id"], require_lock_token=False)
        self.assertIsNone(accepted["final"])
        self.assertEqual(len(accepted["next_clips"]), 1)
        stage2_anchor = db.one("SELECT * FROM clips WHERE id=?", (anchor["id"],))
        self.assertEqual(stage2_anchor["status"], "pending")
        self.assertEqual(stage2_anchor["anchor_stage"], "replace_collector")
        self.assertTrue(stage2_anchor["public_url"].endswith("/accepted/%s/clip_0000_stage1_input.mp4" % uuid))
        self.assertTrue(Path(stage2_anchor["local_path"]).exists())
        episode = db.one("SELECT * FROM episodes WHERE uuid=?", (uuid,))
        self.assertIsNone(episode["anchor_clip_id"])
        self.assertEqual(episode["continuity_state"], "anchor_candidates")
        self.assertEqual(db.one("SELECT COUNT(*) AS count FROM clips WHERE episode_uuid=? AND input_kind='rolling'", (uuid,))["count"], 0)

        stage2_result = queue_rolling_generation(mode="mock")
        self.assertEqual(stage2_result[0]["clip_id"], anchor["id"])
        self.assertEqual(stage2_result[0]["status"], "succeeded")
        accepted = review_clip(anchor["id"], "accept", stage2_result[0]["job_id"], require_lock_token=False)
        self.assertEqual({clip["direction"] for clip in accepted["next_clips"]}, {"backward", "forward"})
        episode = db.one("SELECT * FROM episodes WHERE uuid=?", (uuid,))
        self.assertEqual(episode["anchor_clip_id"], anchor["id"])
        self.assertEqual(episode["continuity_state"], "bidirectional")

        rolling = db.rows("SELECT * FROM clips WHERE episode_uuid=? AND input_kind='rolling' ORDER BY direction", (uuid,))
        self.assertEqual(len(rolling), 2)
        for clip in rolling:
            self.assertEqual(clip["status"], "pending")
            self.assertEqual(clip["source_duration_sec"], 14.0)
            self.assertEqual(clip["overlap_sec"], 1.0)
            self.assertEqual(clip["duration_sec"], 15.0)
            self.assertTrue(Path(clip["local_path"]).exists())

        second = queue_rolling_generation(mode="mock")
        succeeded_ids = {item["clip_id"] for item in second if item.get("status") == "succeeded"}
        self.assertEqual(succeeded_ids, {clip["id"] for clip in rolling})
        forward = db.one("SELECT * FROM clips WHERE episode_uuid=? AND input_kind='rolling' AND direction='forward'", (uuid,))
        forward_job = next(item["job_id"] for item in second if item.get("clip_id") == forward["id"])
        review_clip(forward["id"], "reject", forward_job, require_lock_token=False)
        before_count = db.one("SELECT COUNT(*) AS count FROM clips WHERE episode_uuid=?", (uuid,))["count"]
        rerun = queue_rolling_generation(mode="mock")
        after_count = db.one("SELECT COUNT(*) AS count FROM clips WHERE episode_uuid=?", (uuid,))["count"]
        self.assertEqual(rerun[0]["clip_id"], forward["id"])
        self.assertEqual(before_count, after_count)

    def test_rolling_prefer_input_length_controls_non_anchor_clip_size(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000033"
        save_settings({"mock_async": False, "continuity_prefer_input_sec": 10})
        self.make_head_ready_episode(uuid, 32)
        self.create_anchor_candidates_with_lock(uuid, [14])

        anchor, _ = self.accept_single_anchor_candidate_to_official(uuid)
        self.assertEqual(anchor["duration_sec"], 4.0)

        rolling = db.rows("SELECT * FROM clips WHERE episode_uuid=? AND input_kind='rolling' ORDER BY direction", (uuid,))

        self.assertEqual(len(rolling), 2)
        self.assertEqual({clip["duration_sec"] for clip in rolling}, {10.0})
        self.assertEqual({clip["source_duration_sec"] for clip in rolling}, {9.0})
        self.assertEqual({clip["overlap_sec"] for clip in rolling}, {1.0})

    def test_list_clips_returns_bidirectional_clips_in_timeline_order(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000032"
        save_settings({"mock_async": False})
        self.make_head_ready_episode(uuid, 32)
        self.create_anchor_candidates_with_lock(uuid, [14])
        self.accept_single_anchor_candidate_to_official(uuid)

        clips = [clip for clip in list_clips() if clip["episode_uuid"] == uuid]

        self.assertEqual([clip["direction"] for clip in clips], ["backward", "anchor", "forward"])
        self.assertEqual([clip["input_kind"] for clip in clips], ["rolling", "anchor", "rolling"])
        self.assertEqual(
            [clip["timeline_start_sec"] for clip in clips],
            sorted(clip["timeline_start_sec"] for clip in clips),
        )

    def test_rejecting_accepted_rolling_clip_deletes_future_pending_clip(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000027"
        save_settings({"mock_async": False})
        self.make_head_ready_episode(uuid, 32)
        self.create_anchor_candidates_with_lock(uuid, [4])
        self.accept_single_anchor_candidate_to_official(uuid)
        first_forward = db.one("SELECT * FROM clips WHERE episode_uuid=? AND input_kind='rolling' AND direction='forward'", (uuid,))
        first_forward_result = queue_rolling_generation(mode="mock")
        first_forward_job = next(item["job_id"] for item in first_forward_result if item.get("clip_id") == first_forward["id"])
        review_clip(first_forward["id"], "accept", first_forward_job, require_lock_token=False)
        pending_next = db.one(
            """
            SELECT * FROM clips
            WHERE episode_uuid=? AND input_kind='rolling' AND direction='forward' AND status='pending'
            """,
            (uuid,),
        )
        self.assertIsNotNone(pending_next)

        rejected = review_clip(first_forward["id"], "reject", first_forward_job, require_lock_token=False)

        self.assertEqual(rejected["deleted_future_clip_count"], 1)
        self.assertIsNone(db.one("SELECT * FROM clips WHERE id=?", (pending_next["id"],)))
        self.assertFalse(Path(pending_next["local_path"]).exists())

    def test_rolling_generation_api_endpoint_runs_pending_anchor(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000026"
        save_settings({"mock_async": False})
        self.make_head_ready_episode(uuid, 8)
        self.create_anchor_candidates_with_lock(uuid, [0])

        res = self.client.post("/api/generation/rolling_run", json={"mode": "mock", "operator_id": "client-a"})

        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()[0]["status"], "succeeded")
        clip = db.one("SELECT * FROM clips WHERE episode_uuid=?", (uuid,))
        self.assertEqual(clip["input_kind"], "anchor")
        self.assertEqual(clip["duration_sec"], 4.0)

    def test_rolling_dry_run_keeps_anchor_clip_pending(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000024"
        self.make_reference_files_for(DEFAULT_SETTINGS["reference_images"])
        self.make_head_ready_episode(uuid, 32)
        self.create_anchor_candidates_with_lock(uuid, [14])

        result = queue_rolling_generation(
            mode="seedance",
            dry_run=True,
            prompt="ignored prompt",
            reference_images=["data:image/png;base64,AA=="],
        )
        payload_path = Path(result[0]["output_path"])
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        clip = db.one("SELECT * FROM clips WHERE id=?", (result[0]["clip_id"],))
        image_items = [item for item in payload["content"] if item["type"] == "image_url"]

        self.assertEqual(payload["duration"], 4)
        self.assertEqual(payload["content"][0]["text"], DEFAULT_PROMPT)
        self.assertEqual(len(image_items), 4)
        self.assertEqual(
            [item["image_url"]["url"] for item in image_items],
            [resolve_image_value(item) for item in DEFAULT_SETTINGS["reference_images"]],
        )
        self.assertEqual(clip["status"], "pending")
        self.assertEqual(clip["input_kind"], "anchor")

    def test_anchor_stage_two_and_rolling_use_locked_generation_presets(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000034"
        self.make_reference_files_for(DEFAULT_SETTINGS["reference_images"])
        self.make_reference_files_for(COLLECTOR_ONLY_REFERENCE_IMAGES)
        self.make_reference_files_for(IPHONE2DEPLOY_REFERENCE_IMAGES)
        save_settings({"mock_async": False, "reference_images": ["data:image/png;base64,AA=="], "default_prompt": "custom default"})
        self.make_head_ready_episode(uuid, 32)
        self.create_anchor_candidates_with_lock(uuid, [14])

        first = queue_rolling_generation(mode="mock")
        anchor = db.one("SELECT * FROM clips WHERE episode_uuid=? AND input_kind='anchor'", (uuid,))
        review_clip(anchor["id"], "accept", first[0]["job_id"], require_lock_token=False)

        stage2 = queue_rolling_generation(mode="seedance", dry_run=True, prompt="ignored", reference_images=["data:image/png;base64,AA=="])
        stage2_payload = json.loads(Path(stage2[0]["output_path"]).read_text(encoding="utf-8"))
        stage2_images = [item for item in stage2_payload["content"] if item["type"] == "image_url"]
        self.assertEqual(stage2_payload["content"][0]["text"], COLLECTOR_ONLY_PROMPT)
        self.assertEqual(
            [item["image_url"]["url"] for item in stage2_images],
            [resolve_image_value(item) for item in COLLECTOR_ONLY_REFERENCE_IMAGES],
        )

        stage2_real = queue_rolling_generation(mode="mock")
        review_clip(anchor["id"], "accept", stage2_real[0]["job_id"], require_lock_token=False)
        rolling = db.one("SELECT * FROM clips WHERE episode_uuid=? AND input_kind='rolling' AND direction='forward'", (uuid,))
        self.assertIsNotNone(rolling)

        rolling_dry_run = queue_rolling_generation(mode="seedance", dry_run=True, prompt="ignored", reference_images=["data:image/png;base64,AA=="])
        rolling_payloads = [json.loads(Path(item["output_path"]).read_text(encoding="utf-8")) for item in rolling_dry_run]
        self.assertTrue(rolling_payloads)
        for payload in rolling_payloads:
            image_items = [item for item in payload["content"] if item["type"] == "image_url"]
            self.assertEqual(payload["content"][0]["text"], IPHONE2DEPLOY_PROMPT)
            self.assertEqual(
                [item["image_url"]["url"] for item in image_items],
                [resolve_image_value(item) for item in IPHONE2DEPLOY_REFERENCE_IMAGES],
            )

    def test_bidirectional_stitch_trims_overlap_from_final(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000025"
        save_settings({"mock_async": False})
        self.make_head_ready_episode(uuid, 18)
        self.create_anchor_candidates_with_lock(uuid, [7])

        self.accept_single_anchor_candidate_to_official(uuid)
        rolling_result = queue_rolling_generation(mode="mock")
        for item in rolling_result:
            if item.get("status") == "succeeded":
                review_clip(item["clip_id"], "accept", item["job_id"], require_lock_token=False)

        episode = self.wait_for_final_ready(uuid)
        self.assertEqual(episode["continuity_state"], "complete")
        info = ffprobe_json(Path(episode["final_video_path"])) or ffmpeg_probe_fallback(Path(episode["final_video_path"]))
        self.assertAlmostEqual(float(info["format"]["duration"]), 18.0, delta=0.35)

    def test_partial_continuity_preview_uses_black_for_unaccepted_timeline(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000029"
        save_settings({"mock_async": False})
        self.make_head_ready_episode(uuid, 18)
        self.create_anchor_candidates_with_lock(uuid, [7])

        _, result = self.accept_single_anchor_candidate_to_official(uuid)

        self.assertIsNone(result["final"])
        self.assertEqual(result["preview"]["preview_status"], "stitching")
        episode = self.wait_for_preview_ready(uuid)
        self.assertNotEqual(episode["final_status"], "ready")
        preview = Path(episode["preview_video_path"])
        self.assertTrue(preview.exists())
        info = ffprobe_json(preview) or ffmpeg_probe_fallback(preview)
        self.assertAlmostEqual(float(info["format"]["duration"]), 18.0, delta=0.4)

        clips = db.rows("SELECT * FROM clips WHERE episode_uuid=? ORDER BY timeline_start_sec", (uuid,))
        self.assertEqual([clip["status"] for clip in clips], ["pending", "accepted", "pending"])
        self.assertEqual([clip["input_kind"] for clip in clips], ["rolling", "anchor", "rolling"])

    def test_preview_path_clears_when_no_clips_are_accepted(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000030"
        self.make_head_ready_episode(uuid, 18)
        now = db.now()
        with db.connect() as conn:
            conn.execute(
                """
                UPDATE episodes
                SET preview_video_path=?, preview_status='ready', preview_version=3, updated_at=?
                WHERE uuid=?
                """,
                (str((FINAL_DIR / "stale-preview.mp4").resolve()), now, uuid),
            )

        result = queue_preview_episode(uuid)

        self.assertFalse(result["queued"])
        episode = db.one("SELECT * FROM episodes WHERE uuid=?", (uuid,))
        self.assertIsNone(episode["preview_video_path"])
        self.assertEqual(episode["preview_status"], "missing")
        self.assertEqual(episode["preview_version"], 4)

    def test_preview_video_work_can_cancel_ffmpeg(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ffmpeg cancelled"):
            black_video(DATA_DIR / "cancelled_preview.mp4", 5, should_cancel=lambda: True)

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
        self.assertEqual(settings["generation_presets_version"], GENERATION_PRESETS_VERSION)
        presets = {item["id"]: item for item in settings["generation_presets"]}
        self.assertEqual([item["id"] for item in settings["generation_presets"]], [
            DEFAULT_GENERATION_PRESET_ID,
            COLLECTOR_ONLY_PRESET_ID,
            IPHONE2DEPLOY_PRESET_ID,
        ])
        self.assertEqual(presets["iphone-default"]["prompt"], DEFAULT_PROMPT)
        self.assertEqual(presets["iphone-default"]["reference_images"], settings["reference_images"])
        self.assertEqual(presets["collector-only"]["name"], "仅替换采集器")
        self.assertEqual(presets["collector-only"]["prompt"], COLLECTOR_ONLY_PROMPT)
        self.assertEqual(presets["collector-only"]["reference_images"], COLLECTOR_ONLY_REFERENCE_IMAGES)
        self.assertEqual(presets["iphone2deploy"]["name"], "iphone2deploy")
        self.assertEqual(presets["iphone2deploy"]["prompt"], IPHONE2DEPLOY_PROMPT)
        self.assertEqual(presets["iphone2deploy"]["reference_images"], IPHONE2DEPLOY_REFERENCE_IMAGES)
        self.assertEqual(len(presets["iphone2deploy"]["reference_images"]), 4)

    def test_old_generation_presets_gain_iphone2deploy_once(self) -> None:
        SETTINGS_PATH.write_text(
            json.dumps(
                {
                    "default_prompt": DEFAULT_PROMPT,
                    "reference_images": DEFAULT_SETTINGS["reference_images"],
                    "default_generation_preset_id": "iphone-default",
                    "generation_presets": [
                        {
                            "id": "iphone-default",
                            "name": "iPhone 默认组合",
                            "prompt": DEFAULT_PROMPT,
                            "reference_images": DEFAULT_SETTINGS["reference_images"],
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        settings = load_settings()
        persisted = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        presets = {item["id"]: item for item in settings["generation_presets"]}

        self.assertEqual(settings["generation_presets_version"], GENERATION_PRESETS_VERSION)
        self.assertIn("collector-only", presets)
        self.assertIn("iphone2deploy", presets)
        self.assertEqual(presets["collector-only"]["prompt"], COLLECTOR_ONLY_PROMPT)
        self.assertEqual(presets["collector-only"]["reference_images"], COLLECTOR_ONLY_REFERENCE_IMAGES)
        self.assertEqual(presets["iphone2deploy"]["reference_images"], IPHONE2DEPLOY_REFERENCE_IMAGES)
        self.assertEqual(persisted["generation_presets_version"], GENERATION_PRESETS_VERSION)
        self.assertEqual(len([item for item in persisted["generation_presets"] if item["id"] == "collector-only"]), 1)
        self.assertEqual(len([item for item in persisted["generation_presets"] if item["id"] == "iphone2deploy"]), 1)

        persisted["generation_presets"] = [item for item in persisted["generation_presets"] if item["id"] != "iphone2deploy"]
        SETTINGS_PATH.write_text(json.dumps(persisted, ensure_ascii=False, indent=2), encoding="utf-8")
        settings = load_settings()
        self.assertNotIn("iphone2deploy", {item["id"] for item in settings["generation_presets"]})

    def test_iphone2deploy_refs_are_migrated_to_current_png_images(self) -> None:
        old_refs = [
            "app/reference_images/iphone2deploy-left.jpg",
            "app/reference_images/iphone2deploy-right.jpg",
        ]
        SETTINGS_PATH.write_text(
            json.dumps(
                {
                    "default_prompt": DEFAULT_PROMPT,
                    "reference_images": DEFAULT_SETTINGS["reference_images"],
                    "default_generation_preset_id": "iphone-default",
                    "generation_presets_version": 3,
                    "generation_presets": [
                        {
                            "id": "iphone-default",
                            "name": "iPhone 默认组合",
                            "prompt": DEFAULT_PROMPT,
                            "reference_images": DEFAULT_SETTINGS["reference_images"],
                        },
                        {
                            "id": "iphone2deploy",
                            "name": "iphone2deploy",
                            "prompt": IPHONE2DEPLOY_PROMPT,
                            "reference_images": old_refs,
                        },
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        settings = load_settings()
        persisted = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        presets = {item["id"]: item for item in settings["generation_presets"]}

        self.assertEqual(settings["generation_presets_version"], GENERATION_PRESETS_VERSION)
        self.assertEqual(presets["iphone2deploy"]["reference_images"], IPHONE2DEPLOY_REFERENCE_IMAGES)
        self.assertEqual(
            next(item for item in persisted["generation_presets"] if item["id"] == "iphone2deploy")["reference_images"],
            IPHONE2DEPLOY_REFERENCE_IMAGES,
        )

    def test_iphone2deploy_prompt_and_refs_are_migrated_from_v4(self) -> None:
        old_prompt = "把@视频1中的真人手臂和手机换成@图片1@图片2的机械臂和上面安装的相机，爪夹形态、动作、画面、背景保持不变"
        SETTINGS_PATH.write_text(
            json.dumps(
                {
                    "default_prompt": DEFAULT_PROMPT,
                    "reference_images": DEFAULT_SETTINGS["reference_images"],
                    "default_generation_preset_id": "iphone-default",
                    "generation_presets_version": 4,
                    "generation_presets": [
                        {
                            "id": "iphone-default",
                            "name": "iPhone 默认组合",
                            "prompt": DEFAULT_PROMPT,
                            "reference_images": DEFAULT_SETTINGS["reference_images"],
                        },
                        {
                            "id": "iphone2deploy",
                            "name": "iphone2deploy",
                            "prompt": old_prompt,
                            "reference_images": [
                                "app/reference_images/iphone2deploy-left.png",
                                "app/reference_images/iphone2deploy-right.png",
                            ],
                        },
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        settings = load_settings()
        persisted = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        presets = {item["id"]: item for item in settings["generation_presets"]}

        self.assertEqual(settings["generation_presets_version"], GENERATION_PRESETS_VERSION)
        self.assertEqual(presets["iphone2deploy"]["prompt"], IPHONE2DEPLOY_PROMPT)
        self.assertEqual(presets["iphone2deploy"]["reference_images"], IPHONE2DEPLOY_REFERENCE_IMAGES)
        persisted_preset = next(item for item in persisted["generation_presets"] if item["id"] == "iphone2deploy")
        self.assertEqual(persisted_preset["prompt"], IPHONE2DEPLOY_PROMPT)
        self.assertEqual(persisted_preset["reference_images"], IPHONE2DEPLOY_REFERENCE_IMAGES)

    def test_iphone2deploy_v5_deploy_refs_are_migrated_to_v2(self) -> None:
        old_refs = [
            "app/reference_images/l-near-deploy.png",
            "app/reference_images/r-near-deploy.png",
        ]
        SETTINGS_PATH.write_text(
            json.dumps(
                {
                    "default_prompt": IPHONE2DEPLOY_PROMPT,
                    "reference_images": old_refs,
                    "default_generation_preset_id": "iphone2deploy",
                    "generation_presets_version": 5,
                    "generation_presets": [
                        {
                            "id": "iphone-default",
                            "name": "iPhone default",
                            "prompt": DEFAULT_PROMPT,
                            "reference_images": DEFAULT_SETTINGS["reference_images"],
                        },
                        {
                            "id": "iphone2deploy",
                            "name": "iphone2deploy",
                            "prompt": IPHONE2DEPLOY_PROMPT,
                            "reference_images": old_refs,
                        },
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        settings = load_settings()
        persisted = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        presets = {item["id"]: item for item in settings["generation_presets"]}

        self.assertEqual(settings["generation_presets_version"], GENERATION_PRESETS_VERSION)
        self.assertEqual(settings["reference_images"], IPHONE2DEPLOY_REFERENCE_IMAGES)
        self.assertEqual(presets["iphone2deploy"]["reference_images"], IPHONE2DEPLOY_REFERENCE_IMAGES)
        self.assertEqual(persisted["reference_images"], IPHONE2DEPLOY_REFERENCE_IMAGES)

    def test_iphone2deploy_v6_deploy_refs_are_migrated_to_four_image_prompt(self) -> None:
        old_prompt = "把@视频1里面的真人手臂和手机采集器替换为@图片1@图片2的机械臂和摄像头，爪夹形态、动作、画面、背景保持不变"
        old_refs = [
            "app/reference_images/l-near-deploy-v2.png",
            "app/reference_images/r-near-deploy-v2.png",
        ]
        SETTINGS_PATH.write_text(
            json.dumps(
                {
                    "default_prompt": old_prompt,
                    "reference_images": old_refs,
                    "default_generation_preset_id": "iphone2deploy",
                    "generation_presets_version": 6,
                    "generation_presets": [
                        {
                            "id": "iphone-default",
                            "name": "iPhone default",
                            "prompt": DEFAULT_PROMPT,
                            "reference_images": DEFAULT_SETTINGS["reference_images"],
                        },
                        {
                            "id": "iphone2deploy",
                            "name": "iphone2deploy",
                            "prompt": old_prompt,
                            "reference_images": old_refs,
                        },
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        settings = load_settings()
        persisted = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        presets = {item["id"]: item for item in settings["generation_presets"]}

        self.assertEqual(settings["generation_presets_version"], GENERATION_PRESETS_VERSION)
        self.assertEqual(settings["default_prompt"], IPHONE2DEPLOY_PROMPT)
        self.assertEqual(settings["reference_images"], IPHONE2DEPLOY_REFERENCE_IMAGES)
        self.assertEqual(presets["iphone2deploy"]["prompt"], IPHONE2DEPLOY_PROMPT)
        self.assertEqual(presets["iphone2deploy"]["reference_images"], IPHONE2DEPLOY_REFERENCE_IMAGES)
        self.assertEqual(persisted["default_prompt"], IPHONE2DEPLOY_PROMPT)
        self.assertEqual(persisted["reference_images"], IPHONE2DEPLOY_REFERENCE_IMAGES)

    def test_stale_iphone2deploy_default_is_repaired_even_at_current_version(self) -> None:
        old_prompt = "把@视频1中的真人手臂和手机换成@图片1@图片2的机械臂和上面安装的相机，爪夹形态、动作、画面、背景保持不变"
        old_refs = [
            "app/reference_images/iphone2deploy-left.png",
            "app/reference_images/iphone2deploy-right.png",
        ]
        SETTINGS_PATH.write_text(
            json.dumps(
                {
                    "default_prompt": old_prompt,
                    "reference_images": old_refs,
                    "default_generation_preset_id": "iphone2deploy",
                    "generation_presets_version": GENERATION_PRESETS_VERSION,
                    "generation_presets": [
                        {
                            "id": "iphone-default",
                            "name": "iPhone 默认组合",
                            "prompt": DEFAULT_PROMPT,
                            "reference_images": DEFAULT_SETTINGS["reference_images"],
                        },
                        {
                            "id": "iphone2deploy",
                            "name": "iphone2deploy",
                            "prompt": old_prompt,
                            "reference_images": old_refs,
                        },
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        settings = load_settings()
        persisted = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        presets = {item["id"]: item for item in settings["generation_presets"]}

        self.assertEqual(settings["default_prompt"], IPHONE2DEPLOY_PROMPT)
        self.assertEqual(settings["reference_images"], IPHONE2DEPLOY_REFERENCE_IMAGES)
        self.assertEqual(presets["iphone2deploy"]["prompt"], IPHONE2DEPLOY_PROMPT)
        self.assertEqual(presets["iphone2deploy"]["reference_images"], IPHONE2DEPLOY_REFERENCE_IMAGES)
        self.assertEqual(persisted["default_prompt"], IPHONE2DEPLOY_PROMPT)
        self.assertEqual(persisted["reference_images"], IPHONE2DEPLOY_REFERENCE_IMAGES)

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
            json={"resource_type": "episode", "resource_id": uuid, "owner_id": "alice", "owner_name": "Alice"},
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
                    job_id, clip_id, operator_id, operator_name, api_key_id, api_key_name, status, task_id,
                    model, requested_duration_sec, clip_duration_sec, usage_json,
                    raw_response_json, created_at, updated_at
                )
                VALUES (?, ?, 'client-a', 'Alice', 'key-a', 'Primary key', 'succeeded', 'task-1', 'seedance-fast', 6, 5, ?, ?, ?, ?)
                """,
                (job_cur.lastrowid, clip_id, json.dumps({"total_tokens": 123}), json.dumps({"large": True}), now, now),
            )

        res = self.client.get("/api/usage/seedance")

        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["summary"][0]["operator_id"], "client-a")
        self.assertEqual(data["summary"][0]["call_count"], 1)
        self.assertEqual(data["summary"][0]["requested_duration_sec"], 6)
        self.assertEqual(data["key_summary"][0]["api_key_id"], "key-a")
        self.assertEqual(data["key_summary"][0]["api_key_name"], "Primary key")
        self.assertEqual(data["key_summary"][0]["call_count"], 1)
        self.assertEqual(data["recent_calls"][0]["api_key_name"], "Primary key")
        self.assertEqual(data["recent_calls"][0]["usage"], {"total_tokens": 123})
        self.assertNotIn("raw_response_json", data["recent_calls"][0])

    def test_reviewer_activity_endpoint_summarizes_review_operator(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000021"
        clips, results = self.make_episode_with_generated_clip(uuid, 5.0)
        clip_id = clips[0]["id"]
        lock = self.client.post(
            "/api/locks/acquire",
            json={"resource_type": "episode", "resource_id": uuid, "owner_id": "client-a", "owner_name": "Alice"},
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
            json={"resource_type": "episode", "resource_id": uuid, "owner_id": "alice", "owner_name": "Alice"},
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
        self.assertNotIn("secret-token", json.dumps(data, ensure_ascii=False))
        self.assertNotIn("api_key", data["seedance_api_key_pool"][0])
        self.assertTrue(data["seedance_api_key_pool"][0]["key_set"])

    def test_seedance_api_key_pool_preserves_blank_existing_keys_and_sanitizes_response(self) -> None:
        save_settings(
            {
                "seedance_api_key_pool": [
                    {"id": "key-a", "name": "Primary", "api_key": "secret-a", "concurrency": 2, "enabled": True},
                    {"id": "key-b", "name": "Secondary", "api_key": "secret-b", "concurrency": 1, "enabled": True},
                ]
            }
        )

        res = self.client.post(
            "/api/settings",
            json={
                "values": {
                    "seedance_api_key_pool": [
                        {"id": "key-a", "name": "Primary renamed", "api_key": "", "concurrency": 3, "enabled": False},
                        {"id": "key-c", "name": "Third", "api_key": "secret-c", "concurrency": 1, "enabled": True},
                    ]
                }
            },
        )

        self.assertEqual(res.status_code, 200, res.text)
        public_json = json.dumps(res.json(), ensure_ascii=False)
        self.assertNotIn("secret-a", public_json)
        self.assertNotIn("secret-b", public_json)
        self.assertNotIn("secret-c", public_json)
        settings = load_settings()
        pool = {item["id"]: item for item in settings["seedance_api_key_pool"]}
        self.assertEqual(set(pool), {"key-a", "key-c"})
        self.assertEqual(pool["key-a"]["api_key"], "secret-a")
        self.assertEqual(pool["key-a"]["name"], "Primary renamed")
        self.assertEqual(pool["key-a"]["concurrency"], 3)
        self.assertFalse(pool["key-a"]["enabled"])
        self.assertEqual(pool["key-c"]["api_key"], "secret-c")

    def test_seedance_api_key_pool_submit_empty_clears_legacy_key(self) -> None:
        save_settings({"seedance_api_key": "legacy-secret"})

        save_settings({"seedance_api_key_pool": []})

        settings = load_settings()
        self.assertEqual(settings["seedance_api_key"], "")
        self.assertEqual(settings["seedance_api_key_pool"], [])

    def test_seedance_key_slots_respect_per_key_concurrency(self) -> None:
        settings = {
            "seedance_api_key_pool": [
                {"id": "key-a", "name": "A", "api_key": "secret-a", "concurrency": 1, "enabled": True},
                {"id": "key-b", "name": "B", "api_key": "secret-b", "concurrency": 2, "enabled": True},
                {"id": "key-c", "name": "C", "api_key": "secret-c", "concurrency": 1, "enabled": False},
                {"id": "key-empty", "name": "Empty", "api_key": "", "concurrency": 1, "enabled": True},
            ]
        }

        slots = [backend_services.acquire_seedance_key_slot(settings) for _ in range(3)]
        backend_services.release_seedance_key_slot(None)

        self.assertEqual([slot["id"] for slot in slots], ["key-a", "key-b", "key-b"])
        self.assertEqual(backend_services._SEEDANCE_KEY_ACTIVE, {"key-a": 1, "key-b": 2})
        self.assertEqual(backend_services._GENERATION_ACTIVE, 3)
        for slot in slots:
            backend_services.release_seedance_key_slot(slot)
        self.assertEqual(backend_services._SEEDANCE_KEY_ACTIVE, {})
        self.assertEqual(backend_services._GENERATION_ACTIVE, 0)

    def test_old_mock_timing_is_migrated_to_fast_default(self) -> None:
        SETTINGS_PATH.write_text(json.dumps({"mock_seconds_per_video_second": 24}, ensure_ascii=False), encoding="utf-8")

        settings = load_settings()
        persisted = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))

        self.assertEqual(settings["mock_seconds_per_video_second"], DEFAULT_MOCK_SECONDS_PER_VIDEO_SECOND)
        self.assertEqual(persisted["mock_seconds_per_video_second"], DEFAULT_MOCK_SECONDS_PER_VIDEO_SECOND)

        save_settings({"mock_seconds_per_video_second": 0.02})
        self.assertEqual(load_settings()["mock_seconds_per_video_second"], 0.02)

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

    def test_recover_interrupted_seedance_job_resumes_existing_task(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000035"
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
        self.make_video(head, 5.0)
        clips = create_clips(uuid, head, 5.0)
        with db.connect() as conn:
            conn.execute("UPDATE clips SET status='generating' WHERE id=?", (clips[0]["id"],))
            cur = conn.execute(
                """
                INSERT INTO generation_jobs(
                    clip_id, mode, requested_duration_sec, task_id, status, retry_count,
                    started_at, created_at, updated_at
                )
                VALUES (?, 'seedance', 5, 'cgt-existing', 'running', 0, ?, ?, ?)
                """,
                (clips[0]["id"], now, now, now),
            )
            job_id = cur.lastrowid

        with patch.object(backend_services._GENERATION_EXECUTOR, "submit") as submit:
            recovered = backend_services.recover_interrupted_generation_jobs()

        self.assertEqual(recovered, {"resumed": [job_id], "failed": []})
        submit.assert_called_once_with(backend_services.seedance_job_worker, job_id)
        self.assertEqual(db.one("SELECT status FROM generation_jobs WHERE id=?", (job_id,))["status"], "running")
        self.assertEqual(db.one("SELECT status FROM clips WHERE id=?", (clips[0]["id"],))["status"], "generating")

    def test_recover_interrupted_unsubmitted_job_marks_failed(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000036"
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
        self.make_video(head, 5.0)
        clips = create_clips(uuid, head, 5.0)
        with db.connect() as conn:
            conn.execute("UPDATE clips SET status='generating' WHERE id=?", (clips[0]["id"],))
            cur = conn.execute(
                """
                INSERT INTO generation_jobs(
                    clip_id, mode, requested_duration_sec, status, retry_count,
                    started_at, created_at, updated_at
                )
                VALUES (?, 'mock', 5, 'running', 0, ?, ?, ?)
                """,
                (clips[0]["id"], now, now, now),
            )
            job_id = cur.lastrowid

        recovered = backend_services.recover_interrupted_generation_jobs()

        self.assertEqual(recovered, {"resumed": [], "failed": [job_id]})
        self.assertEqual(db.one("SELECT status FROM generation_jobs WHERE id=?", (job_id,))["status"], "failed")
        self.assertEqual(db.one("SELECT status FROM clips WHERE id=?", (clips[0]["id"],))["status"], "generated_failed")

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

    def test_preprocess_skips_head_ready_episode_without_creating_clips(self) -> None:
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
        lock = self.client.post(
            "/api/locks/acquire",
            json={"resource_type": "episode", "resource_id": uuid, "owner_id": "alice", "owner_name": "Alice"},
        )
        self.assertEqual(lock.status_code, 200, lock.text)
        token = lock.json()["token"]

        skipped = preprocess_one(uuid, load_settings(), fetch_remote=False, lock_token=token)

        self.assertEqual(skipped["status"], "skipped")
        self.assertEqual(skipped["clip_count"], 0)
        self.assertEqual(skipped["clips"], [])
        self.assertIsNone(db.one("SELECT * FROM clips WHERE episode_uuid=?", (uuid,)))
        episode = db.one("SELECT * FROM episodes WHERE uuid=?", (uuid,))
        self.assertEqual(episode["continuity_state"], "select_anchor")

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

        try:
            self.assertTrue(result["queued"])
            submit.assert_called_once()
            locks = db.rows("SELECT * FROM resource_locks WHERE owner_id='system-stitcher'")
            resources = {(row["resource_type"], row["resource_id"]) for row in locks}
            self.assertIn(("episode", uuid), resources)
            for clip in clips:
                self.assertIn(("clip", str(clip["id"])), resources)
        finally:
            with backend_services._STITCH_LOCK:
                backend_services._STITCHING_EPISODES.discard(uuid)
            with db.connect() as conn:
                conn.execute("DELETE FROM resource_locks WHERE owner_id='system-stitcher'")

    def test_episode_lock_required_for_review_api(self) -> None:
        uuid = "00000000-0000-0000-0000-000000000005"
        clips, results = self.make_episode_with_generated_clip(uuid)
        clip_id = clips[0]["id"]
        job_id = results[0]["job_id"]

        missing = self.client.post(f"/api/review/{clip_id}", json={"decision": "accept", "job_id": job_id})
        self.assertEqual(missing.status_code, 423, missing.text)

        lock = self.client.post(
            "/api/locks/acquire",
            json={
                "resource_type": "episode",
                "resource_id": uuid,
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
        self.make_episode_with_generated_clip(uuid)

        first = self.client.post(
            "/api/locks/acquire",
            json={"resource_type": "episode", "resource_id": uuid, "owner_id": "alice", "owner_name": "Alice"},
        )
        self.assertEqual(first.status_code, 200, first.text)
        token = first.json()["token"]

        blocked = self.client.post(
            "/api/locks/acquire",
            json={"resource_type": "episode", "resource_id": uuid, "owner_id": "bob", "owner_name": "Bob"},
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
            json={"resource_type": "episode", "resource_id": uuid, "owner_id": "bob", "owner_name": "Bob"},
        )
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()["owner_name"], "Bob")

    def test_bulk_generation_respects_episode_lock_owner(self) -> None:
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
                "resource_type": "episode",
                "resource_id": uuid,
                "owner_id": "alice",
                "owner_name": "Alice",
            },
        )
        self.assertEqual(locked.status_code, 200, locked.text)

        generated = run_generation(mode="mock")
        self.assertEqual(generated, [])
        statuses = {row["id"]: row["status"] for row in db.rows("SELECT id, status FROM clips")}
        self.assertEqual(statuses[clips[0]["id"]], "pending")
        self.assertEqual(statuses[clips[1]["id"]], "pending")

        generated = run_generation(mode="mock", operator_id="alice", operator_name="Alice")
        self.assertEqual({item["clip_id"] for item in generated}, {clip["id"] for clip in clips})
        statuses = {row["id"]: row["status"] for row in db.rows("SELECT id, status FROM clips")}
        self.assertEqual(statuses[clips[0]["id"]], "generated")
        self.assertEqual(statuses[clips[1]["id"]], "generated")

    def test_episode_lock_blocks_other_operator(self) -> None:
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
        create_clips(uuid, head, 5)
        episode_lock = self.client.post(
            "/api/locks/acquire",
            json={
                "resource_type": "episode",
                "resource_id": uuid,
                "owner_id": "alice",
                "owner_name": "Alice",
            },
        )
        self.assertEqual(episode_lock.status_code, 200, episode_lock.text)
        blocked = self.client.post(
            "/api/locks/acquire",
            json={"resource_type": "episode", "resource_id": uuid, "owner_id": "bob", "owner_name": "Bob"},
        )
        self.assertEqual(blocked.status_code, 409, blocked.text)
        self.assertEqual(blocked.json()["detail"]["lock"]["owner_name"], "Alice")
        res = self.client.post(
            "/api/pipeline/preprocess",
            json={"uuids": [uuid], "fetch_remote": False, "lock_tokens": {uuid: "wrong-token"}},
        )
        self.assertEqual(res.status_code, 409, res.text)
        self.assertEqual(res.json()["detail"]["lock"]["owner_name"], "Alice")


if __name__ == "__main__":
    unittest.main()
