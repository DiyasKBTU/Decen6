"""Этап 4: контекст для LLM, вызов и проверка ответа. При любой проблеме — шаблон этапа 1."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import llm, templates
from .classify import Classification
from .extract import Facts
from .langid import detect
from .policy import HARD_FORBIDDEN, PUNISHMENT_PROMISE, check_citations
from .retrieval import get_history_retriever, get_rule_retriever
from .rules import RuleIndex
from .taxonomy import Taxonomy

_NICK = re.compile(r"\b[A-Z][a-z]{1,15}_[A-Z][a-z]{1,15}\d{0,3}\b")
_GZ_WORDS = re.compile(r"\bзз\b|\bzz\b|\bgz\b|\bгз\b|зел[её]н\w* зон|yashil zona", re.I)
MAX_CANDIDATES = 10


@dataclass
class DraftOutcome:
    draft: templates.Draft
    source: str  # llm | template
    fallback_reason: str | None = None
    hint_reason: str | None = None
    cost_usd: float = 0.0
    latency_ms: float | None = None
    usage: dict = field(default_factory=dict)


def candidate_ids(cls: Classification, facts: Facts, kind: str, text: str, rules: RuleIndex, tax: Taxonomy,
                  evidence=None) -> list[str]:
    """Пункты, на которые модели разрешено сослаться. Решает код (найденное RAG + категория), а не модель."""
    ids: list[str] = []

    def add(pid: str | None) -> None:
        if pid and rules.exists(pid) and pid not in ids:
            ids.append(pid)

    add(cls.primary_ref)
    add(templates.DEFAULT_REFS.get(cls.situation))
    add(templates.SECOND_REFS.get(cls.situation))
    for ref in facts.rule_refs:  # заявленные игроком — только если подходят к категории
        if tax.category_for_ref(ref, kind) == cls.category or cls.category == "complaint.other":
            add(ref)
    if cls.category != "complaint.dm":  # для DM пункт зависит от места (ЗЗ, дом) — его выбрал классификатор
        for ref in tax.rules_of(cls.category):
            add(ref)
    if cls.category in ("complaint.scam", "report.account"):
        add("1.4")
    if cls.situation in ("complaint_violation", "complaint_review"):
        add("ЖБ 1.2.2")
    gz_ok = bool(_GZ_WORDS.search(text))
    hits = evidence.rules if evidence is not None else get_rule_retriever().search(text, k=6)
    for hit in hits:
        point = rules.get(hit.id)
        if point.prefix == "GZ" and not gz_ok:
            continue
        if (point.prefix in ("", "GZ", "ЖБ") and hit.score >= 0.15) or hit.score >= 0.3:
            add(hit.id)
    return ids[:MAX_CANDIDATES]


def _evidence(facts: Facts) -> str:
    parts = []
    if facts.images:
        parts.append(f"скриншотов: {facts.images}")
    if facts.evidence_kind == "video":
        parts.append("есть ссылка на видео")
    elif facts.evidence_kind == "claimed":
        parts.append("игрок пишет, что доказательства есть")
    return ", ".join(parts) or "нет"


def _punishment(cls: Classification, lang: str) -> str | None:
    if cls.situation != "complaint_violation":
        return None
    items = templates.TYPICAL_PUNISHMENT.get(cls.primary_ref or cls.rp_term or "")
    if items:
        return templates.render_punishment(items, lang)
    return "по правилам проекта, конкретную меру определит администратор"


def validate(res: llm.LLMResult, ctx: llm.DraftContext, rules: RuleIndex, allowed_nicks: set[str]) -> str | None:
    """Причина отказа от черновика модели или None, если его можно использовать."""
    text = res.text
    if not 15 <= len(text) <= 1200:
        return "length"
    allowed = {c.id for c in ctx.candidates}
    if any(c not in allowed for c in res.cited):
        return "cited_not_allowed"
    final, problems = check_citations(text, res.cited, rules)
    if problems:
        return "bad_reference"
    if any(pid not in allowed for pid in final):
        return "reference_not_allowed"
    detected = detect(text)[0]
    if ctx.reply_lang in ("ru", "kk", "uz") and detected != ctx.reply_lang:
        return f"wrong_language:{detected}"
    if ctx.reply_lang in ("en", "es", "pt") and detected not in ("en", "es", "pt"):
        return f"wrong_language:{detected}"
    if HARD_FORBIDDEN.search(text):
        return "forbidden_promise"
    if ctx.situation != "complaint_violation" and PUNISHMENT_PROMISE.search(text):
        return "punishment_outside_decision"
    if any(n not in allowed_nicks for n in _NICK.findall(text)):
        return "unknown_nickname"
    return None


def maybe_llm_draft(
    *, kind: str, text: str, cls: Classification, facts: Facts, reply_lang: str, template: templates.Draft,
    rules: RuleIndex, tax: Taxonomy, evidence=None,
) -> DraftOutcome:
    if not llm.enabled():
        return DraftOutcome(template, "template", "llm_disabled")
    if cls.situation not in llm.LLM_SITUATIONS:
        return DraftOutcome(template, "template", "template_is_enough")

    ids = candidate_ids(cls, facts, kind, text, rules, tax, evidence)
    history = get_history_retriever() if evidence is None else None
    ctx = llm.DraftContext(
        kind=kind,
        reply_lang=reply_lang,
        category=cls.category,
        situation=cls.situation,
        player_text=text,
        nick=facts.nickname,
        evidence=_evidence(facts),
        punishment=_punishment(cls, reply_lang),
        candidates=[
            llm.Candidate(pid, templates.ref_phrase(pid, reply_lang, rules), rules.get(pid).short, rules.get(pid).punishment)
            for pid in ids
        ],
        examples=evidence.examples[:3] if evidence is not None else (history.search(text, kind, reply_lang, k=3) if history else []),
        template_text=template.text,
    )
    try:
        res = llm.get_drafter().draft(ctx)
    except llm.LLMError as exc:
        return DraftOutcome(template, "template", f"llm_error:{exc.reason}")

    allowed_nicks = {n for n in [facts.nickname] if n} | set(_NICK.findall(text))
    reason = validate(res, ctx, rules, allowed_nicks)
    common = dict(cost_usd=res.cost_usd or 0.0, latency_ms=round(1000 * res.latency_s, 1), usage=res.usage)
    if reason:
        return DraftOutcome(template, "template", f"rejected:{reason}", **common)

    final_cited, _ = check_citations(res.text, res.cited, rules)
    draft = templates.Draft(
        lang=reply_lang,
        text=res.text,
        cited=final_cited,
        promises_punishment=template.promises_punishment or bool(PUNISHMENT_PROMISE.search(res.text)),
    )
    return DraftOutcome(draft, "llm", None, res.hint_reason if res.needs_human_hint else None, **common)
