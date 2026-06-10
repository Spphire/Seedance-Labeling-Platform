from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from mcap.reader import make_reader
from mcap.writer import Writer
from nmx_msg.Image_pb2 import Image, RGBD
from nmx_msg.Metadata_pb2 import Metadata

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


def set_seedance_collection_mode(metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(metadata)
    metadata["collection_mode"] = "seedance"
    return metadata


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


def count_topic_messages(preprocessed_dir: Path, topic: str) -> int:
    count = 0
    for mcap_path in mcap_files(preprocessed_dir):
        with mcap_path.open("rb") as f:
            reader = make_reader(f)
            for _, _, _ in reader.iter_messages(topics=[topic]):
                count += 1
    return count


def head_rgb_format(preprocessed_dir: Path, topic: str) -> dict[str, Any]:
    for mcap_path in mcap_files(preprocessed_dir):
        with mcap_path.open("rb") as f:
            reader = make_reader(f)
            for schema, _, message in reader.iter_messages(topics=[topic]):
                image = parse_rgb_message(message.data, schema.name if schema else "")
                if image.data or image.cols or image.rows:
                    encoded_format = image.encoded_format or "h264"
                    if encoded_format.lower() not in {"h264", "h.264"}:
                        raise RuntimeError(f"unsupported head rgb encoding for export: {encoded_format}")
                    return {
                        "cols": int(image.cols or 760),
                        "rows": int(image.rows or 570),
                        "channels": int(image.channels or 3),
                        "encoded_format": encoded_format,
                    }
    raise RuntimeError(f"cannot infer head rgb format for {topic}")


def h264_start_codes(data: bytes) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    index = 0
    limit = len(data) - 3
    while index < limit:
        if data[index : index + 4] == b"\x00\x00\x00\x01":
            positions.append((index, 4))
            index += 4
            continue
        if data[index : index + 3] == b"\x00\x00\x01":
            positions.append((index, 3))
            index += 3
            continue
        index += 1
    return positions


def split_h264_access_units(data: bytes) -> list[bytes]:
    starts = h264_start_codes(data)
    aud_positions = [
        position
        for position, size in starts
        if position + size < len(data) and data[position + size] & 0x1F == 9
    ]
    if not aud_positions:
        raise RuntimeError("encoded h264 stream does not contain access unit delimiters")
    chunks = []
    for index, position in enumerate(aud_positions):
        start = 0 if index == 0 and position > 0 else position
        end = aud_positions[index + 1] if index + 1 < len(aud_positions) else len(data)
        chunk = data[start:end]
        if chunk:
            chunks.append(chunk)
    return chunks


def encode_video_access_units(
    final_video: Path,
    frame_count: int,
    fps: float,
    tmp_dir: Path,
    rgb_format: dict[str, Any],
) -> list[bytes]:
    if frame_count <= 0:
        raise RuntimeError("head topic has no frames to replace")
    width = int(rgb_format.get("cols") or 760)
    height = int(rgb_format.get("rows") or 570)
    raw_h264 = tmp_dir / "seedance_final.raw.h264"
    stop_duration = max(1.0, frame_count / max(fps, 1.0) + 1.0)
    run_ffmpeg(
        [
            "-i",
            str(final_video),
            "-vf",
            f"scale={width}:{height},setsar=1,fps={fps:.6f},tpad=stop_mode=clone:stop_duration={stop_duration:.6f}",
            "-frames:v",
            str(frame_count),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-x264-params",
            "aud=1",
            "-f",
            "h264",
            str(raw_h264),
        ]
    )
    units = split_h264_access_units(raw_h264.read_bytes())
    if len(units) < frame_count:
        raise RuntimeError(f"encoded final video produced {len(units)} frames, expected {frame_count}")
    return units[:frame_count]


def update_metadata_payload(data: bytes, frame_count: int) -> bytes:
    message = Metadata()
    message.ParseFromString(data)
    try:
        payload = json.loads(message.metadata.decode("utf-8"))
    except Exception:
        return data
    message.metadata = json.dumps(set_seedance_collection_mode(payload), ensure_ascii=False).encode("utf-8")
    return message.SerializeToString()


def replace_rgb_payload(data: bytes, schema_name: str, frame_data: bytes, rgb_format: dict[str, Any]) -> bytes:
    cols = int(rgb_format.get("cols") or 760)
    rows = int(rgb_format.get("rows") or 570)
    channels = int(rgb_format.get("channels") or 3)
    encoded_format = str(rgb_format.get("encoded_format") or "h264")
    if schema_name == "nmx.msg.RGBD":
        rgbd = RGBD()
        rgbd.ParseFromString(data)
        rgbd.rgb.data = frame_data
        rgbd.rgb.encoded_format = encoded_format
        rgbd.rgb.cols = cols
        rgbd.rgb.rows = rows
        rgbd.rgb.channels = channels
        return rgbd.SerializeToString()
    image = Image()
    image.ParseFromString(data)
    image.data = frame_data
    image.encoded_format = encoded_format
    image.cols = cols
    image.rows = rows
    image.channels = channels
    return image.SerializeToString()


def rewrite_mcap_rgb_topic(
    mcap_path: Path,
    head_topic: str,
    frames: list[bytes],
    start_index: int,
    frame_count: int,
    rgb_format: dict[str, Any],
) -> int:
    temp_path = mcap_path.with_name(f".{mcap_path.name}.seedance-{int(time.time() * 1000)}.mcap")
    schema_ids: dict[int, int] = {}
    channel_ids: dict[int, int] = {}
    frame_index = start_index
    try:
        with mcap_path.open("rb") as source, temp_path.open("wb") as output:
            reader = make_reader(source)
            writer = Writer(output)
            writer.start()
            for schema, channel, message in reader.iter_messages():
                if schema and schema.id not in schema_ids:
                    schema_ids[schema.id] = writer.register_schema(schema.name, schema.encoding, schema.data)
                if channel.id not in channel_ids:
                    schema_id = schema_ids.get(channel.schema_id, 0)
                    channel_ids[channel.id] = writer.register_channel(
                        channel.topic,
                        channel.message_encoding,
                        schema_id,
                        dict(channel.metadata or {}),
                    )
                data = message.data
                schema_name = schema.name if schema else ""
                if channel.topic == head_topic:
                    if frame_index >= len(frames):
                        raise RuntimeError(f"not enough encoded frames for {head_topic}")
                    data = replace_rgb_payload(data, schema_name, frames[frame_index], rgb_format)
                    frame_index += 1
                elif channel.topic == "nmx/nedf/metadata" and schema_name == "nmx.msg.Metadata":
                    data = update_metadata_payload(data, frame_count)
                writer.add_message(
                    channel_ids[channel.id],
                    log_time=message.log_time,
                    publish_time=message.publish_time,
                    sequence=message.sequence,
                    data=data,
                )
            writer.finish()
        temp_path.replace(mcap_path)
        return frame_index
    finally:
        temp_path.unlink(missing_ok=True)


def resolve_episode_root(path: Path) -> Path:
    if (path / "preprocessed" / "metadata.json").exists():
        return path
    if path.name == "preprocessed" and (path / "metadata.json").exists():
        return path.parent
    raise RuntimeError(f"cannot find preprocessed metadata under {path}")


def export_seedance_dataset(source_episode_dir: Path, final_video_path: Path, output_episode_dir: Path) -> dict[str, Any]:
    source_root = resolve_episode_root(source_episode_dir)
    source_preprocessed = source_root / "preprocessed"
    if not final_video_path.exists():
        raise RuntimeError(f"final video does not exist: {final_video_path}")
    metadata = load_json(source_preprocessed / "metadata.json")
    head_topic, camera_id = choose_head_topic(metadata)
    frame_count = count_topic_messages(source_preprocessed, head_topic)
    fps = estimate_fps(source_preprocessed, head_topic)
    rgb_format = head_rgb_format(source_preprocessed, head_topic)
    output_episode_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_root = output_episode_dir.parent / f".{output_episode_dir.name}.exporting-{int(time.time() * 1000)}"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    try:
        shutil.copytree(source_root, temp_root, symlinks=True)
        output_preprocessed = temp_root / "preprocessed"
        metadata_path = output_preprocessed / "metadata.json"
        metadata_path.write_text(
            json.dumps(set_seedance_collection_mode(metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = encode_video_access_units(final_video_path, frame_count, fps, Path(tmpdir), rgb_format)
        next_index = 0
        for mcap_path in mcap_files(output_preprocessed):
            next_index = rewrite_mcap_rgb_topic(mcap_path, head_topic, frames, next_index, frame_count, rgb_format)
        if next_index != frame_count:
            raise RuntimeError(f"replaced {next_index} head frames, expected {frame_count}")
        if output_episode_dir.exists():
            shutil.rmtree(output_episode_dir)
        temp_root.rename(output_episode_dir)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return {
        "output_path": str(output_episode_dir.resolve()),
        "preprocessed_path": str((output_episode_dir / "preprocessed").resolve()),
        "head_topic": head_topic,
        "camera_id": camera_id,
        "frame_count": frame_count,
        "fps": fps,
        "rgb_format": rgb_format,
    }


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


def fetch_episode(host: str, remote_root: str, uuid: str, local_dir: Path) -> Path:
    if local_dir.exists() and (local_dir / "preprocessed" / "metadata.json").exists():
        return local_dir
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    source = Path(remote_root) / uuid
    if source.exists():
        return source

    tmp = local_dir.with_suffix(".partial")
    if tmp.exists():
        shutil.rmtree(tmp)

    remote = f"{host}:{remote_root.rstrip('/')}/{uuid}"
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
    return local_dir
