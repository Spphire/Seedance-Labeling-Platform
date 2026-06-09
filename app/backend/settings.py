from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .paths import CONFIG_DIR, REFERENCE_IMAGES_DIR, ROOT


SETTINGS_PATH = CONFIG_DIR / "settings.json"
SECRET_KEYS = {"seedance_api_key", "seedance_api_key_pool"}
DEFAULT_PUBLIC_BASE_URL = "http://106.14.2.243:18080"
DEFAULT_MOCK_SECONDS_PER_VIDEO_SECOND = 0.2
DEFAULT_PROMPT = (
    "把@视频1中的真人手换成@图片1@图片2的机械臂，"
    "把@视频1中真人手臂换成@图片3@图片4中的机械臂，"
    "爪夹形态、动作、画面、背景保持不变"
)
IPHONE2DEPLOY_PROMPT = (
    "把@视频1里面的真人手臂和手机采集器替换为@图片1@图片2@图片3@图片4的机械臂和摄像头，"
    "夹爪形态、动作、画面、背景保持不变"
)
COLLECTOR_ONLY_PROMPT = (
    "把@视频1里面的左右手机采集器替换为@图片1@图片2中夹爪上安装的摄像头，"
    "机械臂形态、动作、画面、背景保持不变"
)
LEGACY_IPHONE2DEPLOY_PROMPTS = {
    (
        "把@视频1里面的真人手臂和手机采集器替换为@图片1@图片2@图片3@图片4的机械臂和摄像头，"
        "根据爪夹上的绿点对齐形态、动作、画面、背景保持不变"
    ),
    (
        "把@视频1里面的真人手臂和手机采集器替换为@图片1@图片2的机械臂和摄像头，"
        "爪夹形态、动作、画面、背景保持不变"
    ),
    "把@视频1中的真人手臂和手机换成@图片1@图片2的机械臂和上面安装的相机，"
    "爪夹形态、动作、画面、背景保持不变",
}
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
IPHONE2DEPLOY_REFERENCE_IMAGES = [
    "app/reference_images/l-nn-deploy.png",
    "app/reference_images/r-nn-deploy.png",
    "app/reference_images/l-near-deploy-v2.png",
    "app/reference_images/r-near-deploy-v2.png",
]
COLLECTOR_ONLY_REFERENCE_IMAGES = [
    "app/reference_images/l-nn-deploy.png",
    "app/reference_images/r-nn-deploy.png",
]
LEGACY_IPHONE2DEPLOY_REFERENCE_IMAGE_ORDERS = {
    (
        "app/reference_images/l-near-deploy-v2.png",
        "app/reference_images/r-near-deploy-v2.png",
    ),
    (
        "app/reference_images/l-near-deploy.png",
        "app/reference_images/r-near-deploy.png",
    ),
    (
        "app/reference_images/iphone2deploy-left.jpg",
        "app/reference_images/iphone2deploy-right.jpg",
    ),
    (
        "app/reference_images/iphone2deploy-left.png",
        "app/reference_images/iphone2deploy-right.png",
    ),
}
DEFAULT_GENERATION_PRESET_ID = "iphone-default"
COLLECTOR_ONLY_PRESET_ID = "collector-only"
IPHONE2DEPLOY_PRESET_ID = "iphone2deploy"
GENERATION_PRESETS_VERSION = 8
DEFAULT_GENERATION_PRESETS = [
    {
        "id": DEFAULT_GENERATION_PRESET_ID,
        "name": "iPhone 默认组合",
        "prompt": DEFAULT_PROMPT,
        "reference_images": DEFAULT_REFERENCE_IMAGES,
    },
    {
        "id": COLLECTOR_ONLY_PRESET_ID,
        "name": "仅替换采集器",
        "prompt": COLLECTOR_ONLY_PROMPT,
        "reference_images": COLLECTOR_ONLY_REFERENCE_IMAGES,
    },
    {
        "id": IPHONE2DEPLOY_PRESET_ID,
        "name": "iphone2deploy",
        "prompt": IPHONE2DEPLOY_PROMPT,
        "reference_images": IPHONE2DEPLOY_REFERENCE_IMAGES,
    },
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
    "mock_seconds_per_video_second": DEFAULT_MOCK_SECONDS_PER_VIDEO_SECOND,
    "seedance_concurrency": 3,
    "seedance_model": "doubao-seedance-2-0-fast-260128",
    "seedance_base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "seedance_api_key": "",
    "seedance_api_key_pool": [],
    "seedance_resolution": "480p",
    "seedance_ratio": "4:3",
    "seedance_seconds_per_video_second": 24,
    "continuity_overlap_sec": 1,
    "continuity_prefer_input_sec": 12,
    "default_prompt": DEFAULT_PROMPT,
    "reference_images": DEFAULT_REFERENCE_IMAGES,
    "default_generation_preset_id": DEFAULT_GENERATION_PRESET_ID,
    "generation_presets_version": GENERATION_PRESETS_VERSION,
    "generation_presets": DEFAULT_GENERATION_PRESETS,
}


def _env_api_key() -> str:
    return os.environ.get("SEEDANCE_API_KEY") or os.environ.get("ARK_API_KEY") or ""


def _bool_value(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}
    return default


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def _api_key_fingerprint(api_key: str) -> str:
    api_key = api_key.strip()
    if not api_key:
        return ""
    return f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "set"


def _normalize_seedance_api_key_pool(
    value: Any,
    legacy_key: str = "",
    legacy_concurrency: Any = 1,
    include_empty: bool = True,
) -> list[dict[str, Any]]:
    raw = value if isinstance(value, list) else []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        item = item if isinstance(item, dict) else {}
        key_id = str(item.get("id") or f"key-{index + 1}").strip() or f"key-{index + 1}"
        if key_id in seen:
            key_id = f"{key_id}-{index + 1}"
        seen.add(key_id)
        api_key = str(item.get("api_key") or "").strip()
        if not include_empty and not api_key:
            continue
        normalized.append(
            {
                "id": key_id,
                "name": str(item.get("name") or key_id).strip() or key_id,
                "api_key": api_key,
                "concurrency": _positive_int(item.get("concurrency"), _positive_int(legacy_concurrency, 1)),
                "enabled": _bool_value(item.get("enabled"), True),
            }
        )
    if not normalized and legacy_key:
        normalized.append(
            {
                "id": "default",
                "name": "default",
                "api_key": legacy_key.strip(),
                "concurrency": _positive_int(legacy_concurrency, 1),
                "enabled": True,
            }
        )
    return normalized


def _public_seedance_api_key_pool(settings: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": item["id"],
            "name": item["name"],
            "concurrency": item["concurrency"],
            "enabled": item["enabled"],
            "key_set": bool(item.get("api_key")),
            "fingerprint": _api_key_fingerprint(str(item.get("api_key") or "")),
        }
        for item in seedance_api_key_pool(settings, include_empty=True)
    ]


def seedance_api_key_pool(settings: dict[str, Any], include_empty: bool = False) -> list[dict[str, Any]]:
    return _normalize_seedance_api_key_pool(
        settings.get("seedance_api_key_pool"),
        str(settings.get("seedance_api_key") or ""),
        settings.get("seedance_concurrency") or 1,
        include_empty=include_empty,
    )


def _merge_seedance_api_key_pool(existing: dict[str, Any], incoming: Any) -> list[dict[str, Any]]:
    existing_pool = {item["id"]: item for item in seedance_api_key_pool(existing, include_empty=True)}
    result: list[dict[str, Any]] = []
    raw = incoming if isinstance(incoming, list) else []
    for index, item in enumerate(raw):
        item = item if isinstance(item, dict) else {}
        key_id = str(item.get("id") or f"key-{index + 1}").strip() or f"key-{index + 1}"
        existing_item = existing_pool.get(key_id, {})
        api_key = str(item.get("api_key") or "").strip() or str(existing_item.get("api_key") or "")
        result.append(
            {
                "id": key_id,
                "name": str(item.get("name") or existing_item.get("name") or key_id).strip() or key_id,
                "api_key": api_key,
                "concurrency": _positive_int(
                    item.get("concurrency"),
                    existing_item.get("concurrency") or existing.get("seedance_concurrency") or 1,
                ),
                "enabled": _bool_value(item.get("enabled"), _bool_value(existing_item.get("enabled"), True)),
            }
        )
    return _normalize_seedance_api_key_pool(result, include_empty=True)


def _prompt_needs_repair(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return True
    if "\ufffd" in value:
        return True
    return value in LEGACY_DEFAULT_PROMPTS or value in BROKEN_DEFAULT_PROMPTS


def _renamed_reference_images(values: tuple[Any, ...]) -> list[str]:
    return [REFERENCE_IMAGE_RENAMES.get(str(item), str(item)) for item in values]


def _normalized_reference_images(value: Any) -> list[str]:
    refs = tuple(value or [])
    if (not refs or refs in LEGACY_REFERENCE_IMAGE_ORDERS) and REFERENCE_IMAGES_DIR.exists():
        return list(DEFAULT_REFERENCE_IMAGES)
    return _renamed_reference_images(refs)


def _preset_copy(preset: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(preset["id"]),
        "name": str(preset["name"]),
        "prompt": str(preset["prompt"]),
        "reference_images": list(preset["reference_images"]),
    }


def _default_preset_by_id(preset_id: str) -> dict[str, Any] | None:
    return next((preset for preset in DEFAULT_GENERATION_PRESETS if str(preset["id"]) == preset_id), None)


def _migrate_iphone2deploy_preset(preset: dict[str, Any]) -> bool:
    if preset.get("id") != "iphone2deploy":
        return False
    default_preset = _default_preset_by_id("iphone2deploy")
    if not default_preset:
        return False
    refs = tuple(str(item) for item in preset.get("reference_images") or [])
    legacy_prompt = str(preset.get("prompt") or "") in LEGACY_IPHONE2DEPLOY_PROMPTS
    legacy_refs = refs in LEGACY_IPHONE2DEPLOY_REFERENCE_IMAGE_ORDERS
    if not legacy_prompt and not legacy_refs:
        return False
    changed = False
    if preset.get("prompt") != default_preset["prompt"]:
        preset["prompt"] = str(default_preset["prompt"])
        changed = True
    if preset.get("reference_images") != default_preset["reference_images"]:
        preset["reference_images"] = list(default_preset["reference_images"])
        changed = True
    return changed


def _normalize_generation_presets(data: dict[str, Any]) -> bool:
    changed = False
    default_prompt_present = "default_prompt" in data
    reference_images_present = "reference_images" in data
    normalized_default_prompt = data.get("default_prompt")
    if _prompt_needs_repair(normalized_default_prompt):
        normalized_default_prompt = DEFAULT_PROMPT
        changed = True
    normalized_reference_images = _renamed_reference_images(tuple(data.get("reference_images") or [])) if reference_images_present else None
    if reference_images_present and (not data.get("reference_images") or tuple(data.get("reference_images") or []) in LEGACY_REFERENCE_IMAGE_ORDERS) and REFERENCE_IMAGES_DIR.exists():
        normalized_reference_images = list(DEFAULT_REFERENCE_IMAGES)
        changed = True
    raw_presets = data.get("generation_presets")
    if not isinstance(raw_presets, list) or not raw_presets:
        raw_presets = [_preset_copy(preset) for preset in DEFAULT_GENERATION_PRESETS]
        raw_presets[0]["prompt"] = normalized_default_prompt
        raw_presets[0]["reference_images"] = normalized_reference_images or list(DEFAULT_REFERENCE_IMAGES)
        changed = True

    presets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_presets):
        item = item if isinstance(item, dict) else {}
        preset_id = str(item.get("id") or f"preset-{index + 1}").strip() or f"preset-{index + 1}"
        if preset_id in seen_ids:
            preset_id = f"{preset_id}-{index + 1}"
            changed = True
        seen_ids.add(preset_id)
        name = str(item.get("name") or preset_id).strip() or preset_id
        prompt = item.get("prompt")
        if _prompt_needs_repair(prompt):
            prompt = DEFAULT_PROMPT
            changed = True
        refs = _normalized_reference_images(item.get("reference_images"))
        normalized = {
            "id": preset_id,
            "name": name,
            "prompt": str(prompt),
            "reference_images": refs,
        }
        if normalized != item:
            changed = True
        if _migrate_iphone2deploy_preset(normalized):
            changed = True
        presets.append(normalized)

    try:
        presets_version = int(data.get("generation_presets_version") or 1)
    except (TypeError, ValueError):
        presets_version = 1
    if presets_version < GENERATION_PRESETS_VERSION:
        for preset in DEFAULT_GENERATION_PRESETS:
            if str(preset["id"]) in seen_ids:
                continue
            copied = _preset_copy(preset)
            presets.append(copied)
            seen_ids.add(copied["id"])
            changed = True
        if presets_version < 4:
            for preset in presets:
                if preset["id"] != "iphone2deploy":
                    continue
                default_preset = _default_preset_by_id("iphone2deploy")
                if default_preset and preset["reference_images"] != default_preset["reference_images"]:
                    preset["reference_images"] = list(default_preset["reference_images"])
                    changed = True
        if presets_version < 5:
            for preset in presets:
                if preset["id"] != "iphone2deploy":
                    continue
                default_preset = _default_preset_by_id("iphone2deploy")
                if not default_preset:
                    continue
                if preset["prompt"] != default_preset["prompt"]:
                    preset["prompt"] = str(default_preset["prompt"])
                    changed = True
                if preset["reference_images"] != default_preset["reference_images"]:
                    preset["reference_images"] = list(default_preset["reference_images"])
                    changed = True
        if presets_version < 6:
            for preset in presets:
                if preset["id"] != "iphone2deploy":
                    continue
                default_preset = _default_preset_by_id("iphone2deploy")
                if default_preset and preset["reference_images"] != default_preset["reference_images"]:
                    preset["reference_images"] = list(default_preset["reference_images"])
                    changed = True
        if presets_version < 7:
            for preset in presets:
                if preset["id"] != "iphone2deploy":
                    continue
                default_preset = _default_preset_by_id("iphone2deploy")
                if not default_preset:
                    continue
                if preset["prompt"] != default_preset["prompt"]:
                    preset["prompt"] = str(default_preset["prompt"])
                    changed = True
                if preset["reference_images"] != default_preset["reference_images"]:
                    preset["reference_images"] = list(default_preset["reference_images"])
                    changed = True
        data["generation_presets_version"] = GENERATION_PRESETS_VERSION
        changed = True
    elif data.get("generation_presets_version") != GENERATION_PRESETS_VERSION:
        data["generation_presets_version"] = GENERATION_PRESETS_VERSION
        changed = True

    required_order = [str(preset["id"]) for preset in DEFAULT_GENERATION_PRESETS]
    ordered_presets: list[dict[str, Any]] = []
    for preset_id in required_order:
        existing = next((preset for preset in presets if preset["id"] == preset_id), None)
        if existing:
            ordered_presets.append(existing)
    ordered_presets.extend(preset for preset in presets if preset["id"] not in required_order)
    if [preset["id"] for preset in ordered_presets] != [preset["id"] for preset in presets]:
        presets = ordered_presets
        changed = True

    default_id = str(data.get("default_generation_preset_id") or "").strip()
    if default_id not in seen_ids:
        default_id = presets[0]["id"]
        changed = True
    default_preset = next((item for item in presets if item["id"] == default_id), presets[0])
    if default_preset["id"] == "iphone2deploy":
        if normalized_default_prompt in LEGACY_IPHONE2DEPLOY_PROMPTS:
            normalized_default_prompt = default_preset["prompt"]
            changed = True
        if (
            reference_images_present
            and normalized_reference_images is not None
            and tuple(normalized_reference_images) in LEGACY_IPHONE2DEPLOY_REFERENCE_IMAGE_ORDERS
        ):
            normalized_reference_images = list(default_preset["reference_images"])
            changed = True
    if default_prompt_present and default_preset["prompt"] != normalized_default_prompt:
        default_preset["prompt"] = str(normalized_default_prompt)
        changed = True
    if reference_images_present:
        assert normalized_reference_images is not None
        if default_preset["reference_images"] != normalized_reference_images:
            default_preset["reference_images"] = normalized_reference_images
            changed = True
    else:
        normalized_reference_images = list(default_preset["reference_images"])
    normalized_default_prompt = str(default_preset["prompt"])

    data["generation_presets"] = presets
    data["default_generation_preset_id"] = default_id
    if data.get("default_prompt") != normalized_default_prompt:
        data["default_prompt"] = normalized_default_prompt
        changed = True
    if data.get("reference_images") != normalized_reference_images:
        data["reference_images"] = normalized_reference_images
        changed = True
    return changed


def _settings_from_disk() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(DEFAULT_SETTINGS, ensure_ascii=False, indent=2), encoding="utf-8")
        return dict(DEFAULT_SETTINGS)
    data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    changed = False
    try:
        mock_seconds = float(data.get("mock_seconds_per_video_second", DEFAULT_MOCK_SECONDS_PER_VIDEO_SECOND))
    except (TypeError, ValueError):
        mock_seconds = DEFAULT_MOCK_SECONDS_PER_VIDEO_SECOND
    if mock_seconds > DEFAULT_MOCK_SECONDS_PER_VIDEO_SECOND:
        data["mock_seconds_per_video_second"] = DEFAULT_MOCK_SECONDS_PER_VIDEO_SECOND
        changed = True
    if _normalize_generation_presets(data):
        changed = True
    normalized_pool = _normalize_seedance_api_key_pool(
        data.get("seedance_api_key_pool"),
        str(data.get("seedance_api_key") or ""),
        data.get("seedance_concurrency") or 1,
        include_empty=True,
    )
    if data.get("seedance_api_key_pool") != normalized_pool:
        data["seedance_api_key_pool"] = normalized_pool
        changed = True
    if changed:
        SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    merged = dict(DEFAULT_SETTINGS)
    merged.update(data)
    return merged


def load_settings() -> dict[str, Any]:
    settings = _settings_from_disk()
    env_key = _env_api_key()
    if env_key and not settings.get("seedance_api_key") and not seedance_api_key_pool(settings):
        settings["seedance_api_key"] = env_key
        settings["seedance_api_key_pool"] = _normalize_seedance_api_key_pool(
            [],
            env_key,
            settings.get("seedance_concurrency") or 1,
            include_empty=True,
        )
    return settings


def save_settings(data: dict[str, Any]) -> dict[str, Any]:
    merged = _settings_from_disk()
    incoming = dict(data)
    pool_submitted = "seedance_api_key_pool" in incoming
    if "seedance_api_key_pool" in incoming:
        incoming["seedance_api_key_pool"] = _merge_seedance_api_key_pool(merged, incoming["seedance_api_key_pool"])
    if "seedance_api_key" in incoming and incoming["seedance_api_key"] and "seedance_api_key_pool" not in incoming:
        incoming["seedance_api_key_pool"] = _merge_seedance_api_key_pool(
            merged,
            [
                {
                    "id": "default",
                    "name": "default",
                    "api_key": incoming["seedance_api_key"],
                    "concurrency": incoming.get("seedance_concurrency", merged.get("seedance_concurrency", 1)),
                    "enabled": True,
                }
            ],
        )
    merged.update(incoming)
    if pool_submitted:
        pool = _normalize_seedance_api_key_pool(
            merged.get("seedance_api_key_pool"),
            "",
            merged.get("seedance_concurrency") or 1,
            include_empty=True,
        )
        merged["seedance_api_key_pool"] = pool
    else:
        pool = seedance_api_key_pool(merged, include_empty=True)
    first_key = next((str(item.get("api_key") or "") for item in pool if item.get("api_key")), "")
    if pool_submitted or first_key:
        merged["seedance_api_key"] = first_key
    SETTINGS_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_settings()


def public_settings() -> dict[str, Any]:
    settings = load_settings()
    visible = {key: value for key, value in settings.items() if key not in SECRET_KEYS}
    public_pool = _public_seedance_api_key_pool(settings)
    visible["seedance_api_key_pool"] = public_pool
    visible["seedance_api_key_set"] = any(item["key_set"] for item in public_pool)
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
