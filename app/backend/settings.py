from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .paths import CONFIG_DIR, REFERENCE_IMAGES_DIR, ROOT


SETTINGS_PATH = CONFIG_DIR / "settings.json"
SECRET_KEYS = {"seedance_api_key"}
DEFAULT_PUBLIC_BASE_URL = "http://106.14.2.243:18080"
DEFAULT_PROMPT = (
    "把@视频1中的真人手换成@图片1@图片2的机械臂，"
    "把@视频1中真人手臂换成@图片3@图片4中的机械臂，"
    "爪夹形态、动作、画面、背景保持不变"
)
LEGACY_DEFAULT_PROMPTS = {
    "保持参考视频中的视角方向、背景、动作和时序连续性，生成与输入 clip 时长一致的视频。",
}
BROKEN_DEFAULT_PROMPTS = {
    "?@??1???????@??1@??2??????@??1???????@??3@??4???????????????????????",
}
DEFAULT_REFERENCE_IMAGES = [
    "app/reference_images/l-near-iphone.png",
    "app/reference_images/r-near-iphone.png",
    "app/reference_images/l-far-iphone.png",
    "app/reference_images/r-far-iphone.png",
]
REFERENCE_IMAGE_RENAMES = {
    "app/reference_images/l-near.png": "app/reference_images/l-near-iphone.png",
    "app/reference_images/r-near.png": "app/reference_images/r-near-iphone.png",
    "app/reference_images/l-far.png": "app/reference_images/l-far-iphone.png",
    "app/reference_images/r-far.png": "app/reference_images/r-far-iphone.png",
}
LEGACY_REFERENCE_IMAGE_ORDERS = {
    (
        "app/reference_images/l-far.png",
        "app/reference_images/l-near.png",
        "app/reference_images/r-far.png",
        "app/reference_images/r-near.png",
    ),
    (
        "app/reference_images/l-far.png",
        "app/reference_images/r-far.png",
        "app/reference_images/l-near.png",
        "app/reference_images/r-near.png",
    ),
}


DEFAULT_SETTINGS: dict[str, Any] = {
    "dm3_host": "DM3data",
    "dm3_nedf_root": "/mnt/nm_data/data/nedf",
    "public_base_url": DEFAULT_PUBLIC_BASE_URL,
    "generation_mode": "mock",
    "mock_concurrency": 8,
    "mock_async": True,
    "mock_seconds_per_video_second": 24,
    "seedance_concurrency": 3,
    "seedance_model": "doubao-seedance-2-0-fast-260128",
    "seedance_base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "seedance_api_key": "",
    "seedance_resolution": "480p",
    "seedance_ratio": "4:3",
    "seedance_seconds_per_video_second": 24,
    "default_prompt": DEFAULT_PROMPT,
    "reference_images": DEFAULT_REFERENCE_IMAGES,
}


def _env_api_key() -> str:
    return os.environ.get("SEEDANCE_API_KEY") or os.environ.get("ARK_API_KEY") or ""


def _prompt_needs_repair(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return True
    if "\ufffd" in value:
        return True
    return value in LEGACY_DEFAULT_PROMPTS or value in BROKEN_DEFAULT_PROMPTS


def _renamed_reference_images(values: tuple[Any, ...]) -> list[str]:
    return [REFERENCE_IMAGE_RENAMES.get(str(item), str(item)) for item in values]


def _settings_from_disk() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(DEFAULT_SETTINGS, ensure_ascii=False, indent=2), encoding="utf-8")
        return dict(DEFAULT_SETTINGS)
    data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    changed = False
    if _prompt_needs_repair(data.get("default_prompt")):
        data["default_prompt"] = DEFAULT_PROMPT
        changed = True
    refs = tuple(data.get("reference_images") or [])
    if (not refs or refs in LEGACY_REFERENCE_IMAGE_ORDERS) and REFERENCE_IMAGES_DIR.exists():
        data["reference_images"] = DEFAULT_REFERENCE_IMAGES
        changed = True
    elif refs:
        renamed_refs = _renamed_reference_images(refs)
        if renamed_refs != list(refs):
            data["reference_images"] = renamed_refs
            changed = True
    if changed:
        SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    merged = dict(DEFAULT_SETTINGS)
    merged.update(data)
    return merged


def load_settings() -> dict[str, Any]:
    settings = _settings_from_disk()
    env_key = _env_api_key()
    if env_key and not settings.get("seedance_api_key"):
        settings["seedance_api_key"] = env_key
    return settings


def save_settings(data: dict[str, Any]) -> dict[str, Any]:
    merged = _settings_from_disk()
    merged.update(data)
    SETTINGS_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_settings()


def public_settings() -> dict[str, Any]:
    settings = load_settings()
    visible = {key: value for key, value in settings.items() if key not in SECRET_KEYS}
    visible["seedance_api_key_set"] = bool(settings.get("seedance_api_key"))
    visible["available_reference_images"] = available_reference_images()
    return visible


def available_reference_images() -> list[dict[str, str]]:
    if not REFERENCE_IMAGES_DIR.exists():
        return []
    paths = {path.resolve().relative_to(ROOT).as_posix(): path for path in REFERENCE_IMAGES_DIR.iterdir()}
    ordered_keys = [key for key in DEFAULT_REFERENCE_IMAGES if key in paths]
    ordered_keys.extend(sorted(key for key in paths if key not in ordered_keys))
    result = []
    for key in ordered_keys:
        path = paths[key]
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        result.append(
            {
                "id": key,
                "name": path.stem,
                "url": f"/reference_images/{path.name}",
            }
        )
    return result


def public_url_for(static_kind: str, relative_path: Path) -> str:
    settings = load_settings()
    base = str(settings["public_base_url"]).rstrip("/")
    rel = relative_path.as_posix()
    return f"{base}/{static_kind}/{rel}"
