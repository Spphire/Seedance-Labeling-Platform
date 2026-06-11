from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Literal

from fastapi import HTTPException, Request


Role = Literal["admin", "reviewer"]


@dataclass(frozen=True)
class Principal:
    role: Role
    token_name: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on", "enabled"}


def dev_mode_enabled() -> bool:
    return _truthy_env("SEEDANCE_DEV_MODE")


def auth_tokens_configured() -> bool:
    return any(
        os.environ.get(name, "").strip()
        for name in ["SEEDANCE_ADMIN_TOKEN", "SEEDANCE_REVIEWER_TOKEN"]
    )


def security_enabled() -> bool:
    return _truthy_env("SEEDANCE_REQUIRE_AUTH") or auth_tokens_configured()


def _bearer_token(request: Request) -> str:
    header = request.headers.get("authorization", "").strip()
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-seedance-token", "").strip()


def _token_matches(candidate: str, expected: str) -> bool:
    return bool(candidate and expected and secrets.compare_digest(candidate, expected))


def authenticate_request(request: Request) -> Principal:
    if not security_enabled():
        return Principal(role="admin", token_name="dev")
    token = _bearer_token(request)
    admin_token = os.environ.get("SEEDANCE_ADMIN_TOKEN", "").strip()
    reviewer_token = os.environ.get("SEEDANCE_REVIEWER_TOKEN", "").strip()
    if _token_matches(token, admin_token):
        return Principal(role="admin", token_name="admin")
    if _token_matches(token, reviewer_token):
        return Principal(role="reviewer", token_name="reviewer")
    raise HTTPException(status_code=401, detail="authentication token is required")


def require_reviewer(request: Request) -> Principal:
    return authenticate_request(request)


def require_admin(request: Request) -> Principal:
    principal = authenticate_request(request)
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="admin permission is required")
    return principal

