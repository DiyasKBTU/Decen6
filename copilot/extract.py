"""Извлечение фактов из обращения. Это заявления игрока, а не установленные факты."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .rules import RuleIndex

_RE_NICK = re.compile(r"\b[A-Z][a-z]{1,15}_[A-Z][a-z]{1,15}\d{0,3}\b")
_RE_SERVER = re.compile(
    r"(?:сервер[а-я]*|server|serv|срв|сер)\s*(?:№|#|n)?\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:-?й\s*)?(?:сервер|server)",
    re.I,
)
_RE_VIDEO = re.compile(r"<url>|фрапс|fraps|видео|video|запис|ютуб|youtube|youtu|rutube|vk\.com|tiktok|ролик", re.I)
_RE_SCREEN_WORDS = re.compile(r"скрин|screen|скриншот|док-?ва|доказательств|dalil|rasm", re.I)
_RE_NO_EVIDENCE = re.compile(r"(нет|без)\s+(скрин|док|видео|фрапс)", re.I)


@dataclass
class Facts:
    nickname: str | None
    rule_refs: list[str]
    has_evidence: bool
    evidence_kind: str | None  # video | screenshot | claimed | None
    server: int | None
    images: int
    notes: list[str] = field(default_factory=list)


def extract(
    player_text: str,
    rules: RuleIndex,
    *,
    reported_nick: str | None = None,
    rule_refs_raw: str | None = None,
    images: int = 0,
) -> Facts:
    nickname = reported_nick or None
    if not nickname:
        m = _RE_NICK.search(player_text)
        nickname = m.group(0) if m else None

    refs = rules.parse_refs(f"{rule_refs_raw or ''} \n {player_text}")

    if _RE_VIDEO.search(player_text):
        kind = "video"
    elif images > 0:
        kind = "screenshot"
    elif _RE_SCREEN_WORDS.search(player_text) and not _RE_NO_EVIDENCE.search(player_text):
        kind = "claimed"
    else:
        kind = None

    server = None
    m = _RE_SERVER.search(player_text)
    if m:
        value = int(m.group(1) or m.group(2))
        server = value if 0 < value < 200 else None

    return Facts(
        nickname=nickname,
        rule_refs=refs,
        has_evidence=kind in ("video", "screenshot"),
        evidence_kind=kind,
        server=server,
        images=images,
    )
