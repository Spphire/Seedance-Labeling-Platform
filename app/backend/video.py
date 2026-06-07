from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

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


def run_ffmpeg_cancellable(args: list[str], should_cancel: Callable[[], bool] | None = None) -> None:
    if should_cancel is None:
        run_ffmpeg(args)
        return
    cmd = [ffmpeg_path(), "-hide_banner", "-y", *args]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout = ""
    stderr = ""
    try:
        while True:
            if should_cancel():
                proc.terminate()
                try:
                    stdout, stderr = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    stdout, stderr = proc.communicate()
                raise RuntimeError("ffmpeg cancelled")
            try:
                stdout, stderr = proc.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                continue
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{stderr}")
    except Exception:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
        raise


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
                "-vf",
                "scale=760:570,setsar=1,fps=30",
                "-movflags",
                "+faststart",
                str(dst),
            ]
        )
    finally:
        list_path.unlink(missing_ok=True)


def concat_videos_precise(inputs: list[Path], dst: Path, should_cancel: Callable[[], bool] | None = None) -> None:
    if not inputs:
        raise ValueError("inputs must not be empty")
    dst.parent.mkdir(parents=True, exist_ok=True)
    args: list[str] = []
    for path in inputs:
        args.extend(["-i", str(path)])
    filter_parts = [
        f"[{index}:v]scale=760:570,setsar=1,setpts=PTS-STARTPTS[v{index}]"
        for index in range(len(inputs))
    ]
    filter_parts.append(
        "".join(f"[v{index}]" for index in range(len(inputs)))
        + f"concat=n={len(inputs)}:v=1:a=0,fps=30,format=yuv420p[v]"
    )
    run_ffmpeg_cancellable(
        [
            *args,
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dst),
        ],
        should_cancel,
    )


def trim_video(
    src: Path,
    dst: Path,
    start_sec: float,
    duration_sec: float,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg_cancellable(
        [
            "-i",
            str(src),
            "-ss",
            f"{start_sec:.6f}",
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
        ],
        should_cancel,
    )


def black_video(
    dst: Path,
    duration_sec: float,
    width: int = 760,
    height: int = 570,
    fps: int = 30,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg_cancellable(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:size={width}x{height}:rate={fps}",
            "-t",
            f"{duration_sec:.6f}",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dst),
        ],
        should_cancel,
    )


def compose_rolling_input(
    head_src: Path,
    dst: Path,
    source_start_sec: float,
    source_duration_sec: float,
    anchor_src: Path | None = None,
    overlap_sec: float = 0.0,
) -> None:
    if source_duration_sec <= 0:
        raise ValueError("source_duration_sec must be positive")
    if overlap_sec <= 0:
        cut_clip(head_src, dst, source_start_sec, source_duration_sec)
        return
    if anchor_src is None:
        raise ValueError("anchor_src is required when overlap_sec is positive")
    anchor_duration = video_duration(anchor_src)
    if anchor_duration + 0.05 < overlap_sec:
        raise ValueError(f"anchor video is shorter than overlap: {anchor_duration:.3f}s < {overlap_sec:.3f}s")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tail_path = tmp / "anchor_tail.mp4"
        source_path = tmp / "source_window.mp4"
        trim_video(anchor_src, tail_path, max(0.0, anchor_duration - overlap_sec), overlap_sec)
        cut_clip(head_src, source_path, source_start_sec, source_duration_sec)
        concat_videos_precise([tail_path, source_path], dst)


def rolling_clip_plan(
    total_duration: float,
    overlap_sec: int = 1,
    min_input_sec: int = 4,
    max_input_sec: int = 15,
) -> list[dict[str, float | int]]:
    planned_total = int(math.floor(total_duration + 1e-6))
    if planned_total <= 0:
        return []
    if planned_total < min_input_sec:
        raise ValueError(f"Cannot make rolling clips for {total_duration:.3f}s within [{min_input_sec}, {max_input_sec}]")
    if overlap_sec < 0:
        raise ValueError("overlap_sec must be non-negative")
    first_max = max_input_sec
    next_max = max_input_sec - overlap_sec
    next_min = max(1, min_input_sec - overlap_sec)
    if next_max < next_min:
        raise ValueError("overlap_sec leaves no room for source video")

    timeline_durations: list[int] = []
    remaining = planned_total
    first = min(first_max, remaining)
    timeline_durations.append(first)
    remaining -= first
    while remaining > 0:
        chunk = min(next_max, remaining)
        timeline_durations.append(chunk)
        remaining -= chunk

    if len(timeline_durations) >= 2 and timeline_durations[-1] < next_min:
        borrow = next_min - timeline_durations[-1]
        timeline_durations[-2] -= borrow
        timeline_durations[-1] += borrow

    for index, timeline_duration in enumerate(timeline_durations):
        if index == 0:
            min_timeline = min_input_sec
            max_timeline = first_max
        else:
            min_timeline = next_min
            max_timeline = next_max
        if timeline_duration < min_timeline or timeline_duration > max_timeline:
            raise ValueError(f"Cannot make rolling clips for {total_duration:.3f}s within [{min_input_sec}, {max_input_sec}]")

    plan: list[dict[str, float | int]] = []
    source_start = 0
    for index, timeline_duration in enumerate(timeline_durations):
        overlap = 0 if index == 0 else overlap_sec
        duration = timeline_duration + overlap
        plan.append(
            {
                "clip_index": index,
                "start_sec": float(source_start),
                "duration_sec": float(duration),
                "source_start_sec": float(source_start),
                "source_duration_sec": float(timeline_duration),
                "overlap_sec": float(overlap),
                "timeline_duration_sec": float(timeline_duration),
            }
        )
        source_start += timeline_duration
    return plan


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
