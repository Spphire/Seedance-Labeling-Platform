from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .ids import normalize_uuid


class EpisodeBatchRequest(BaseModel):
    episodes_text: str = Field(..., description="One episode UUID per line or separated by whitespace/commas.")


class SubmitPreprocessRequest(EpisodeBatchRequest):
    fetch_remote: bool = True
    lock_tokens: dict[str, str] | None = None


class PreprocessRequest(BaseModel):
    uuids: list[str] | None = None
    fetch_remote: bool = True
    lock_tokens: dict[str, str] | None = None

    @field_validator("uuids")
    @classmethod
    def normalize_uuids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        return [normalize_uuid(item) for item in value]


class GenerationRunRequest(BaseModel):
    mode: str | None = None
    clip_ids: list[int] | None = None
    dry_run: bool = False
    lock_token: str | None = None


class ReviewRequest(BaseModel):
    decision: str
    job_id: int | None = None
    note: str = ""
    lock_token: str | None = None


class ImportHeadVideoRequest(BaseModel):
    uuid: str
    path: str
    lock_token: str | None = None

    @field_validator("uuid")
    @classmethod
    def normalize_uuid(cls, value: str) -> str:
        return normalize_uuid(value)


class LockRequest(BaseModel):
    resource_type: str
    resource_id: str
    owner_id: str
    owner_name: str = ""
    ttl_sec: int | None = None
    force: bool = False


class LockRenewRequest(BaseModel):
    token: str
    owner_id: str
    ttl_sec: int | None = None


class LockReleaseRequest(BaseModel):
    token: str
    owner_id: str


class LockTokenRequest(BaseModel):
    lock_token: str | None = None
    mode: str | None = None
