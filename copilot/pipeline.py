"""Сборка всех шагов: обращение -> ответ по контракту.

Ответы админов (role=admin) никогда не используются: на скрытом тесте их не будет,
а при оценке это была бы утечка правильного ответа.
"""

from __future__ import annotations

from typing import Any

from . import config, dedup, rag, template_library, templates
from .classify import classify, refine_with_model
from .drafting import DraftOutcome, maybe_llm_draft
from .extract import extract
from .langid import detect
from .ml import get_models, rule_features
from .normalize import normalize, to_game_encoding
from .policy import decide
from .rules import get_index
from .schemas import DraftReply, Extracted, TriageRequest, TriageResponse
from .taxonomy import get_taxonomy

PLAYER_ROLES = {"player"}


def _player_text(req: TriageRequest) -> tuple[str, int, bool]:
    texts, attachments, accused_replied = [], 0, False
    for m in req.messages:
        if m.role in PLAYER_ROLES:
            texts.append(m.text)
            attachments += m.attachments
        elif m.role == "reported_player":
            accused_replied = True
    return "\n".join(texts), attachments, accused_replied


_AUTO = object()


def triage(req: TriageRequest, debug: bool = False, dup: Any = _AUTO) -> tuple[TriageResponse, dict[str, Any]]:
    """-> (ответ по контракту, служебные данные: источник черновика, стоимость, ...).

    dup: результат поиска дублей, если он уже посчитан (пакетный прогон идёт по времени);
    по умолчанию обращение проверяется и добавляется в историю здесь же.
    """
    if dup is _AUTO:
        dup = dedup.check_and_add(req.model_dump()) if dedup.enabled() else None
    rules, tax = get_index(), get_taxonomy()
    raw_text, attachments, accused_replied = _player_text(req)
    text, norm_flags = normalize(raw_text)
    lang, lang_conf, reply_lang = detect(text, req.server_lang)

    images = req.images_count if req.images_count is not None else attachments
    facts = extract(text, rules, reported_nick=req.reported_nick, rule_refs_raw=req.rule_refs_raw, images=images)
    cls = classify(req.kind, text, facts, rules, tax, rule_refs_raw=req.rule_refs_raw or "")
    rule_category = cls.category
    models = get_models()
    if models and req.kind in models:
        feats = rule_features(req.kind, cls, facts, lang, accused_replied, text)
        pred = models[req.kind].predict(text, req.rule_refs_raw or "", feats)
        cls = refine_with_model(cls, req.kind, text, facts, rules, tax, pred.category, pred.prob, config.ML_MIN_PROB)
    if dup is not None:
        # категория и приоритет остаются: обращение по-прежнему оценивается само по себе
        cls.situation = f"{req.kind}_duplicate"
        cls.primary_ref = None
        cls.is_spam = True
        cls.reasons.insert(0, f"повтор обращения {dup.of_ticket or 'без id'} ({dup.minutes_ago} мин назад, "
                              f"сходство {dup.similarity:.2f}), первое ещё без ответа")

    # RAG: перед любым черновиком находим пункты правил и похожие ответы админов
    evidence = rag.gather(req.kind, text, req.rule_refs_raw, cls, reply_lang, rules, tax)
    if config.RAG_CHOOSE and cls.situation == "complaint_violation":
        cls.primary_ref = rag.choose_citation(cls, evidence)

    template = templates.build(
        cls.situation, reply_lang, rules,
        nick=facts.nickname, ref=cls.primary_ref, rp_term=cls.rp_term, category=cls.category,
        urgent=cls.threat, self_harm=cls.self_harm,
    )
    source = None
    library = template_library.library_draft(req.kind, reply_lang, cls.situation, raw_text, req.rule_refs_raw)
    if library:
        # шаблон, который админ принял во вкладке «Шаблоны», вместо общего ответа
        tpl, match = library
        template = templates.Draft(lang=reply_lang, text=tpl["text"],
                                   cited=[p for p in rules.parse_refs(tpl["text"]) if rules.exists(p)])
        cls.situation, source = "library", "library"
        cls.confidence = max(cls.confidence, 0.7)  # ответ для этого кластера уже одобрил админ
        cls.reasons.insert(0, f"принятый шаблон {tpl['suggestion_id']} для похожих обращений (сходство {match.similarity:.2f})")
    if source:
        outcome = DraftOutcome(template, source)
    else:
        outcome = maybe_llm_draft(
            kind=req.kind, text=text, cls=cls, facts=facts, reply_lang=reply_lang, template=template,
            rules=rules, tax=tax, evidence=evidence,
        )
    draft = outcome.draft
    if outcome.source == "llm":  # служебные приписки шаблона сохраняем и для текста модели
        if cls.threat:
            draft.text += templates.T["urgent"][draft.lang]
        if cls.self_harm:
            draft.text += templates.T["self_harm"][draft.lang]

    policy = decide(
        category=cls.category, situation=cls.situation, confidence=cls.confidence, text=draft.text,
        cited=draft.cited, promises_punishment=draft.promises_punishment, threat=cls.threat,
        self_harm=cls.self_harm, rules=rules, tax=tax,
    )
    reasons = list(dict.fromkeys(policy.reasons + cls.reasons))
    needs_human = policy.needs_human
    if outcome.hint_reason:
        reasons.append(f"модель просит проверить: {outcome.hint_reason}")
        needs_human = True
    if accused_replied:
        reasons.append("есть комментарий обвиняемого — учесть при решении")
    confidence = round(min(cls.confidence, 0.5 + lang_conf / 2), 2)

    response = TriageResponse(
        lang=lang,
        category=cls.category,
        priority=cls.priority,
        is_spam_or_duplicate=cls.is_spam,
        extracted=Extracted(
            nickname=facts.nickname, rule_refs=facts.rule_refs, has_evidence=facts.has_evidence, server=facts.server,
        ),
        draft_reply=DraftReply(lang=draft.lang, text=draft.text, cited=policy.cited),
        needs_human=needs_human,
        confidence=confidence,
        reasons=reasons,
    )
    meta: dict[str, Any] = {
        "draft_source": outcome.source,
        "llm_fallback_reason": outcome.fallback_reason,
        "llm_cost_usd": outcome.cost_usd,
        "llm_latency_ms": outcome.latency_ms,
    }
    meta["category_source"] = "model" if cls.category != rule_category else "rules"
    meta["duplicate_of"] = dup.of_ticket if dup else None
    meta["evidence"] = evidence.to_meta(policy.cited, rules)
    if debug:
        meta.update({
            "situation": cls.situation,
            "rule_category": rule_category,
            "normalized_text": text,
            "normalization": norm_flags,
            "evidence_kind": facts.evidence_kind,
            "policy_problems": policy.violations,
            "reply_lang": reply_lang,
            "llm_usage": outcome.usage,
            "template_text": template.text if outcome.source == "llm" else None,
        })
        if draft.lang == "kk":
            meta["text_game_encoding"] = to_game_encoding(draft.text)
    elif config.KK_GAME_ENCODING and draft.lang == "kk" and req.kind == "report":
        response.draft_reply.text = to_game_encoding(response.draft_reply.text)
    return response, meta
