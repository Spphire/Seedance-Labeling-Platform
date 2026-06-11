from __future__ import annotations

import base64
import json
import math
import mimetypes
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from volcenginesdkarkruntime import Ark

from app.backend.paths import REFERENCE_IMAGES_DIR, ROOT


def image_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}"


def resolve_image_value(value: str) -> str:
    if value.startswith(("http://", "https://", "data:")):
        raise ValueError("reference image must be selected from the project library")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("reference image must stay inside app/reference_images")
    if not path.is_absolute():
        path = ROOT / path
    try:
        path.resolve().relative_to(REFERENCE_IMAGES_DIR.resolve())
    except ValueError as exc:
        raise ValueError("reference image must stay inside app/reference_images") from exc
    if not path.is_file():
        raise FileNotFoundError(f"reference image not found: {value}")
    return image_uri(path)


class SeedanceClient:
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings

    def mock_generate(self, clip_path: Path, output_path: Path) -> dict[str, Any]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clip_path, output_path)
        return {"task_id": f"mock-{int(time.time() * 1000)}", "output_url": "", "output_path": str(output_path)}

    def dry_run_payload(
        self,
        prompt: str,
        public_url: str,
        duration_sec: float,
        reference_images: list[str] | None = None,
    ) -> dict[str, Any]:
        duration = int(math.ceil(duration_sec))
        content = [{"type": "text", "text": prompt}]
        refs = self.settings.get("reference_images", []) if reference_images is None else reference_images
        for item in refs:
            uri = resolve_image_value(str(item))
            content.append({"type": "image_url", "image_url": {"url": uri}, "role": "reference_image"})
        content.append({"type": "video_url", "video_url": {"url": public_url}, "role": "reference_video"})
        return {
            "model": self.settings["seedance_model"],
            "content": content,
            "resolution": self.settings["seedance_resolution"],
            "ratio": self.settings["seedance_ratio"],
            "duration": duration,
            "generate_audio": False,
            "watermark": False,
        }

    def generate(
        self,
        prompt: str,
        public_url: str,
        duration_sec: float,
        output_path: Path,
        reference_images: list[str] | None = None,
    ) -> dict[str, Any]:
        api_key = self.settings.get("seedance_api_key")
        if not api_key:
            raise RuntimeError("seedance_api_key is required for seedance mode")
        payload = self.dry_run_payload(prompt, public_url, duration_sec, reference_images)
        client = Ark(base_url=self.settings["seedance_base_url"], api_key=api_key)
        result = client.content_generation.tasks.create(**payload)
        task_id = result.id
        return self.wait_and_download(client, task_id, output_path, input_url=public_url)

    def create_task(
        self,
        prompt: str,
        public_url: str,
        duration_sec: float,
        reference_images: list[str] | None = None,
    ) -> dict[str, Any]:
        api_key = self.settings.get("seedance_api_key")
        if not api_key:
            raise RuntimeError("seedance_api_key is required for seedance mode")
        payload = self.dry_run_payload(prompt, public_url, duration_sec, reference_images)
        client = Ark(base_url=self.settings["seedance_base_url"], api_key=api_key)
        result = client.content_generation.tasks.create(**payload)
        data = self._model_dump(result)
        return {"task_id": result.id, "usage": self._find_usage(data), "raw_response": data}

    def wait_for_task(
        self,
        task_id: str,
        output_path: Path,
        input_url: str | None = None,
        on_poll: Callable[[Any], None] | None = None,
        on_download_progress: Callable[[int, int | None], None] | None = None,
    ) -> dict[str, Any]:
        api_key = self.settings.get("seedance_api_key")
        if not api_key:
            raise RuntimeError("seedance_api_key is required for seedance mode")
        client = Ark(base_url=self.settings["seedance_base_url"], api_key=api_key)
        return self.wait_and_download(
            client,
            task_id,
            output_path,
            input_url=input_url,
            on_poll=on_poll,
            on_download_progress=on_download_progress,
        )

    def wait_and_download(
        self,
        client: Ark,
        task_id: str,
        output_path: Path,
        input_url: str | None = None,
        on_poll: Callable[[Any], None] | None = None,
        on_download_progress: Callable[[int, int | None], None] | None = None,
    ) -> dict[str, Any]:
        while True:
            task = client.content_generation.tasks.get(task_id=task_id)
            if on_poll:
                on_poll(task)
            if task.status == "succeeded":
                data = self._model_dump(task)
                output_url = self._find_output_url(data, input_urls={input_url} if input_url else set())
                if not output_url:
                    raise RuntimeError(f"Seedance succeeded but no output URL found: {data}")
                self._download_with_retries(output_url, output_path, on_progress=on_download_progress)
                return {
                    "task_id": task_id,
                    "output_url": output_url,
                    "output_path": str(output_path),
                    "usage": self._find_usage(data),
                    "raw_response": data,
                }
            if task.status == "failed":
                data = self._model_dump(task)
                raise RuntimeError(str(data.get("error") or "Seedance task failed"))
            time.sleep(10)

    @staticmethod
    def _download_with_retries(
        url: str,
        output_path: Path,
        attempts: int = 5,
        timeout_sec: float = 120,
        on_progress: Callable[[int, int | None], None] | None = None,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            tmp_path = output_path.with_name(f".{output_path.name}.{int(time.time() * 1000)}.{attempt}.download")
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "seedance-labeling-platform/1.0"})
                with urllib.request.urlopen(request, timeout=timeout_sec) as response, tmp_path.open("wb") as out:
                    expected = int(response.headers.get("Content-Length") or 0) or None
                    received = 0
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        received += len(chunk)
                        if on_progress:
                            on_progress(received, expected)
                if expected is not None and received < expected:
                    raise RuntimeError(f"download incomplete: got only {received} out of {expected} bytes")
                tmp_path.replace(output_path)
                return
            except Exception as exc:
                last_error = exc
                tmp_path.unlink(missing_ok=True)
                if attempt < attempts:
                    time.sleep(min(2 * attempt, 10))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _model_dump(value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        if isinstance(value, dict):
            return value
        return json.loads(json.dumps(value, default=str))

    @classmethod
    def _find_output_url(cls, data: dict[str, Any], input_urls: set[str] | None = None) -> str:
        input_urls = input_urls or set()
        candidates: list[str] = []

        def add(value: Any) -> None:
            if isinstance(value, str):
                candidates.append(value)

        content = data.get("content")
        if isinstance(content, dict):
            for key in ["video_url", "file_url", "output_url", "url"]:
                add(content.get(key))
        for key in ["output_video_url", "video_url", "file_url", "output_url", "url"]:
            add(data.get(key))

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
            else:
                add(value)

        walk(data)
        for url in candidates:
            if cls._is_output_url(url, input_urls):
                return url
        return ""

    @staticmethod
    def _find_usage(data: dict[str, Any]) -> Any:
        for key in ["usage", "usage_info", "token_usage", "billing", "cost"]:
            value = data.get(key)
            if value:
                return value
        return None

    @staticmethod
    def _is_output_url(value: str, input_urls: set[str]) -> bool:
        if not value.startswith(("http://", "https://")):
            return False
        return value not in input_urls
