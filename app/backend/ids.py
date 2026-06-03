from __future__ import annotations

import re


UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def parse_uuids(text: str) -> list[str]:
    found = UUID_RE.findall(text)
    seen = set()
    result = []
    for item in found:
        lower = item.lower()
        if lower not in seen:
            seen.add(lower)
            result.append(lower)
    return result


def normalize_uuid(value: str) -> str:
    parsed = parse_uuids(value)
    if len(parsed) != 1 or parsed[0] != value.lower():
        raise ValueError(f"invalid uuid: {value}")
    return parsed[0]
