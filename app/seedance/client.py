from __future__ import annotations

import base64
import json
import math
import mimetypes
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any

from volcenginesdkarkruntime import Ark


def image_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}"


class SeedanceClient:
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings

    def mock_generate(self, clip_path: Path, output_path: Path) -> dict[str, Any]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clip_path, output_path)
        return {"task_id": f"mock-{int(time.time() * 1000)}", "output_url": "", "output_path": str(output_path)}

    def dry_run_payload(self, prompt: str, public_url: str, duration_sec: float) -> dict[str, Any]:
        duration = int(math.ceil(duration_sec))
        content = [{"type": "text", "text": prompt}]
        for item in self.settings.get("reference_images", []):
            path = Path(item)
            uri = item if str(item).startswith(("http://", "https://", "data:")) else image_uri(path)
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

    def generate(self, prompt: str, public_url: str, duration_sec: float, output_path: Path) -> dict[str, Any]:
        api_key = self.settings.get("seedance_api_key")
        if not api_key:
            raise RuntimeError("seedance_api_key is required for seedance mode")
        payload = self.dry_run_payload(prompt, public_url, duration_sec)
        client = Ark(base_url=self.settings["seedance_base_url"], api_key=api_key)
        result = client.content_generation.tasks.create(**payload)
        task_id = result.id
        while True:
            task = client.content_generation.tasks.get(task_id=task_id)
            if task.status == "succeeded":
                data = task.model_dump() if hasattr(task, "model_dump") else task.dict()
                output_url = self._find_output_url(data)
                if not output_url:
                    raise RuntimeError(f"Seedance succeeded but no output URL found: {data}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(output_url, output_path)
                return {"task_id": task_id, "output_url": output_url, "output_path": str(output_path)}
            if task.status == "failed":
                raise RuntimeError(str(getattr(task, "error", "Seedance task failed")))
            time.sleep(10)

    @staticmethod
    def _find_output_url(data: dict[str, Any]) -> str:
        text = json.dumps(data, ensure_ascii=False)
        for marker in ["http://", "https://"]:
            idx = text.find(marker)
            if idx >= 0:
                end = min([p for p in [text.find('"', idx), text.find("\\", idx)] if p >= 0] or [len(text)])
                return text[idx:end]
        return ""

