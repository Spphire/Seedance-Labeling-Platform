from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import imageio_ffmpeg


def ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args: list[str]) -> None:
    cmd = [ffmpeg_path(), "-hide_banner", "-y", *args]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{proc.stderr}")


def ffprobe_json(path: Path) -> dict[str, Any]:
    ffmpeg = Path(ffmpeg_path())
    probe = ffmpeg.with_name("ffprobe.exe" if ffmpeg.suffix.lower() == ".exe" else "ffprobe")
    if not probe.exists():
        probe = shutil.which("ffprobe")
    if not probe:
        return {}
    proc = subprocess.run(
        [
            str(probe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        return {}
    return json.loads(proc.stdout)


def ffmpeg_probe_fallback(path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [ffmpeg_path(), "-hide_banner", "-i", str(path), "-f", "null", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = proc.stderr
    duration = None
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if match:
        hours, minutes, seconds = match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    stream: dict[str, Any] = {}
    video_line = next((line for line in text.splitlines() if "Video:" in line), "")
    codec_match = re.search(r"Video:\s*([^,\s]+)", video_line)
    size_match = re.search(r"(\d{2,5})x(\d{2,5})", video_line)
    fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", video_line)
    if codec_match:
        stream["codec_name"] = codec_match.group(1).lower()
    if size_match:
        stream["width"] = int(size_match.group(1))
        stream["height"] = int(size_match.group(2))
    if fps_match:
        fps = float(fps_match.group(1))
        stream["avg_frame_rate"] = f"{int(round(fps))}/1" if abs(fps - round(fps)) < 1e-6 else f"{fps}/1"
    return {"format": {"duration": str(duration or 0)}, "streams": [stream] if stream else []}


def video_duration(path: Path) -> float:
    info = ffprobe_json(path) or ffmpeg_probe_fallback(path)
    try:
        return float(info["format"]["duration"])
    except Exception as exc:
        raise RuntimeError(f"Cannot read duration for {path}") from exc


def transcode_760x570(src: Path, dst: Path, fps: float | None = None) -> None:
    args = ["-i", str(src), "-vf", "scale=760:570,setsar=1"]
    if fps:
        args += ["-r", f"{fps:.6f}"]
    args += ["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(args)


def cut_clip(src: Path, dst: Path, start_sec: float, duration_sec: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-ss",
            f"{start_sec:.6f}",
            "-i",
            str(src),
            "-t",
            f"{duration_sec:.6f}",
            "-vf",
            "scale=760:570,setsar=1",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dst),
        ]
    )


def normalize_accepted(src: Path, dst: Path, duration_sec: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-t",
            f"{duration_sec:.6f}",
            "-vf",
            "scale=760:570,setsar=1,fps=30",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dst),
        ]
    )


def stitch_videos(inputs: list[Path], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
        list_path = Path(f.name)
        for path in inputs:
            escaped = str(path.resolve()).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    try:
        run_ffmpeg(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "30",
                "-vf",
                "scale=760:570,setsar=1",
                "-movflags",
                "+faststart",
                str(dst),
            ]
        )
    finally:
        list_path.unlink(missing_ok=True)


def clip_plan(total_duration: float, min_sec: float = 4.0, max_sec: float = 15.0) -> list[tuple[float, float]]:
    if total_duration <= 0:
        return []
    if total_duration < min_sec:
        raise ValueError(f"Cannot split {total_duration:.3f}s into clips within [{min_sec}, {max_sec}]")
    if total_duration <= max_sec:
        return [(0.0, total_duration)]

    durations: list[float] = []
    remaining = total_duration
    while remaining > max_sec:
        durations.append(max_sec)
        remaining -= max_sec
    if remaining > 0:
        durations.append(remaining)

    if len(durations) >= 2 and durations[-1] < min_sec:
        borrow = math.ceil(min_sec - durations[-1])
        durations[-2] -= borrow
        durations[-1] += borrow
    if any(d < min_sec - 1e-6 or d > max_sec + 1e-6 for d in durations):
        # This can only happen for pathological totals below min_sec; keep a clear error.
        raise ValueError(f"Cannot split {total_duration:.3f}s into clips within [{min_sec}, {max_sec}]")

    result: list[tuple[float, float]] = []
    start = 0.0
    for duration in durations:
        result.append((start, duration))
        start += duration
    return result


def requested_seedance_duration(duration_sec: float) -> int:
    return int(math.ceil(duration_sec))
