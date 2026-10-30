"""Контракт POST /triage (как в брифе) + необязательные поля из выгрузки."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = "player"  # player | reported_player | admin
    text: str = ""
    attachments: int = 0
    ts: str | None = None


class TriageRequest(BaseModel):
    """Минимум по контракту — kind и messages. Остальное есть в jsonl-выгрузке и помогает."""

    model_config = ConfigDict(extra="ignore")

    kind: Literal["complaint", "report"]
    messages: list[Message] = Field(default_factory=list)
    ticket_id: str | None = None
    created_at: str | None = None
    region: str | None = None
    server_lang: str | None = None
    reporter_hash: str | None = None
    reported_hash: str | None = None
    reported_nick: str | None = None
    rule_refs_raw: str | None = None
    images_count: int | None = None


class Extracted(BaseModel):
    nickname: str | None
    rule_refs: list[str]
    has_evidence: bool
    server: int | None


class DraftReply(BaseModel):
    lang: str
    text: str
    cited: list[str]


class TriageResponse(BaseModel):
    lang: str
    category: str
    priority: str
    is_spam_or_duplicate: bool
    extracted: Extracted
    draft_reply: DraftReply
    needs_human: bool
    confidence: float
    reasons: list[str]
