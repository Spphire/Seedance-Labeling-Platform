from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from mcap.reader import make_reader
from nmx_msg.Image_pb2 import Image, RGBD

from .video import run_ffmpeg


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def choose_head_topic(metadata: dict) -> tuple[str, str]:
    for camera_id, alias in (metadata.get("camera_info") or {}).items():
        if "head" in str(alias).lower() or "ego" in str(alias).lower():
            return f"nmx/hal/camera/{camera_id}/rgbd", str(camera_id)
    raise RuntimeError("No head/ego camera found in metadata.camera_info")


def mcap_files(preprocessed_dir: Path) -> list[Path]:
    index_path = preprocessed_dir / "data" / "mcap_index.json"
    if index_path.exists():
        files = []
        for item in load_json(index_path):
            filename = item.get("filename")
            if filename:
                path = preprocessed_dir / "data" / filename
                if path.exists():
                    files.append((item.get("start_time_ns", 0), path))
        if files:
            return [path for _, path in sorted(files, key=lambda pair: pair[0])]
    return sorted((preprocessed_dir / "data").glob("*.mcap"))


def estimate_fps(preprocessed_dir: Path, topic: str) -> float:
    timestamps_path = preprocessed_dir / "timestamps.json"
    if timestamps_path.exists():
        values = load_json(timestamps_path).get(topic)
        if isinstance(values, list) and len(values) >= 2:
            duration = (int(values[-1]) - int(values[0])) / 1e9
            if duration > 0:
                return (len(values) - 1) / duration
    return 30.0


def parse_rgb_message(data: bytes, schema_name: str) -> Image:
    if schema_name == "nmx.msg.RGBD":
        rgbd = RGBD()
        rgbd.ParseFromString(data)
        return rgbd.rgb
    image = Image()
    image.ParseFromString(data)
    if image.data:
        return image
    rgbd = RGBD()
    rgbd.ParseFromString(data)
    if rgbd.rgb.data:
        return rgbd.rgb
    return image


def extract_head_video(preprocessed_dir: Path, output_mp4: Path) -> dict:
    metadata = load_json(preprocessed_dir / "metadata.json")
    topic, camera_id = choose_head_topic(metadata)
    fps = estimate_fps(preprocessed_dir, topic)
    raw_h264 = output_mp4.with_suffix(".raw.h264")
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    raw_h264.unlink(missing_ok=True)
    output_mp4.unlink(missing_ok=True)

    frame_count = 0
    width = height = None
    first_time = last_time = None
    with raw_h264.open("wb") as out:
        for mcap_path in mcap_files(preprocessed_dir):
            with mcap_path.open("rb") as f:
                reader = make_reader(f)
                for schema, _, message in reader.iter_messages(topics=[topic]):
                    image = parse_rgb_message(message.data, schema.name if schema else "")
                    if not image.data:
                        continue
                    if image.encoded_format.lower() not in {"h264", "h.264"}:
                        raise RuntimeError(f"Unsupported image encoding: {image.encoded_format}")
                    out.write(image.data)
                    width = image.cols or width
                    height = image.rows or height
                    frame_count += 1
                    if first_time is None:
                        first_time = message.log_time
                    last_time = message.log_time

    if frame_count == 0:
        raise RuntimeError(f"No frames decoded for {topic}")

    run_ffmpeg(
        [
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(raw_h264),
            "-vf",
            "scale=760:570,setsar=1",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_mp4),
        ]
    )
    raw_h264.unlink(missing_ok=True)
    duration_sec = (last_time - first_time) / 1e9 if first_time and last_time else frame_count / fps
    return {
        "topic": topic,
        "camera_id": camera_id,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "source_width": width,
        "source_height": height,
        "output_path": str(output_mp4),
    }


def fetch_episode(host: str, remote_root: str, uuid: str, local_dir: Path) -> None:
    if local_dir.exists() and (local_dir / "preprocessed" / "metadata.json").exists():
        return
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    remote = f"{host}:{remote_root.rstrip('/')}/{uuid}"
    tmp = local_dir.with_suffix(".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    cmd = ["scp", "-O", "-r", remote, str(tmp)]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"scp failed for {uuid}: {proc.stderr}")
    if local_dir.exists():
        shutil.rmtree(local_dir)
    tmp.rename(local_dir)
