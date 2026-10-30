"""Этап 3: логика смешивания модели и правил (без файла модели) и работа пайплайна с моделью."""

import pytest

from copilot import config
from copilot.classify import Classification, classify, refine_with_model
from copilot.extract import extract
from copilot.ml import get_models
from copilot.pipeline import triage
from copilot.rules import get_index
from copilot.schemas import TriageRequest
from copilot.taxonomy import get_taxonomy

RULES, TAX = get_index(), get_taxonomy()


def rule_decision(kind, text, refs=""):
    facts = extract(text, RULES, rule_refs_raw=refs, images=1)
    return classify(kind, text, facts, RULES, TAX, refs), facts


def refine(kind, text, category, prob, refs=""):
    c, facts = rule_decision(kind, text, refs)
    return c, refine_with_model(c, kind, text, facts, RULES, TAX, category, prob, 0.5)


def test_model_overrides_unclear_rule_and_situation_follows():
    rules_c, new = refine("complaint", "проверьте логи, я ничего не делал", "complaint.admin", 0.9)
    assert new.category == "complaint.admin" and new.situation == "complaint_appeal"
    assert any("классификатор" in r for r in new.reasons)


def test_low_confidence_keeps_rules():
    c, new = refine("complaint", "оск родни", "complaint.dm", 0.45)
    assert new is c


def test_protected_rare_class_needs_high_confidence():
    c, new = refine("complaint", "читер летает с аимом, софт", "complaint.insult", 0.7)
    assert c.category == "complaint.cheat" and new is c
    _, strong = refine("complaint", "читер летает с аимом, софт", "complaint.insult", 0.85)
    assert strong.category == "complaint.insult"


def test_locked_rules_are_final():
    for kind, text in [("report", "оплатил донат, не пришло"), ("report", "+"), ("complaint", "."), ("report", "что такое мг")]:
        c, new = refine(kind, text, "report.complaint" if kind == "report" else "complaint.insult", 0.99)
        assert new is c, text


def test_report_override_to_offtopic_needs_strong_confidence():
    c, new = refine("report", "убил без причины у банка", "report.offtopic", 0.7)
    assert new is c
    _, help_ = refine("report", "убил без причины у банка", "report.help", 0.7)
    assert help_.category == "report.help" and help_.situation == "report_help"


def test_pipeline_without_model(monkeypatch):
    monkeypatch.setattr(config, "ML_MODE", "0")
    get_models.cache_clear()
    try:
        _, meta = triage(TriageRequest.model_validate({"kind": "complaint", "messages": [{"role": "player", "text": "оск"}]}))
        assert meta["category_source"] == "rules"
    finally:
        get_models.cache_clear()


@pytest.mark.skipif(not config.CLASSIFIER_PATH.exists(), reason="нет artifacts/classifier.pkl (make train)")
def test_pipeline_with_model_appeal():
    body = {"kind": "complaint", "messages": [{"role": "player", "text": "проверьте логи, я ничего не делал, где доказательства?"}]}
    resp, meta = triage(TriageRequest.model_validate(body), debug=True)
    assert resp.category == "complaint.admin" and resp.needs_human
