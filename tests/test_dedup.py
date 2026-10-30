"""Бонус: дубли. Логика окна, «ещё без ответа», цели и сходства + работа в пайплайне."""

import uuid

from copilot.dedup import DedupStore, Record, parse_ts
from copilot.pipeline import triage
from copilot.schemas import TriageRequest

T0 = parse_ts("2026-09-01T12:00:00")


def rec(minutes, kind="complaint", target="t1", text="оск родни в чате", answered_after=None, tid=None):
    created = T0 + 60 * minutes
    answered = created + 60 * answered_after if answered_after is not None else None
    return Record(created, tid or uuid.uuid4().hex[:8], kind, target, text, answered)


def test_repeat_while_first_pending_is_duplicate():
    store = DedupStore()
    first = rec(0, tid="A")
    assert store.check_and_add("r", first) is None
    dup = store.check_and_add("r", rec(20, text="оск родни в чате!!"))
    assert dup and dup.of_ticket == "A" and dup.minutes_ago == 20


def test_repeat_after_answer_is_new_submission():
    store = DedupStore()
    store.check_and_add("r", rec(0, answered_after=5))  # ответили через 5 минут
    assert store.check_and_add("r", rec(30)) is None


def test_other_target_or_other_text_is_not_duplicate():
    store = DedupStore()
    store.check_and_add("r", rec(0, target="t1"))
    assert store.check_and_add("r", rec(10, target="t2")) is None
    assert store.check_and_add("r", rec(15, target="t1", text="продаёт аккаунт за реальные деньги в телеграм")) is None


def test_old_complaint_outside_window():
    store = DedupStore()
    store.check_and_add("r", rec(0))
    assert store.check_and_add("r", rec(25 * 60)) is None


def test_report_flood_and_window():
    store = DedupStore()
    store.check_and_add("r", rec(0, kind="report", target=None, text="помогите застрял в текстурах"))
    assert store.check_and_add("r", rec(5, kind="report", target=None, text="помогите застрял в текстурах пж"))
    assert store.check_and_add("r", rec(50, kind="report", target=None, text="помогите застрял в текстурах")) is None


def test_only_earlier_tickets_count():
    store = DedupStore()
    store.check_and_add("r", rec(30, tid="later"))
    assert store.check_and_add("r", rec(0, tid="earlier")) is None


def body(reporter, minutes, text, kind="complaint", target="Arman_Petrov"):
    return TriageRequest.model_validate({
        "kind": kind, "reporter_hash": reporter, "reported_nick": target if kind == "complaint" else None,
        "created_at": f"2026-09-01T12:{minutes:02d}:00", "ticket_id": uuid.uuid4().hex[:10],
        "messages": [{"role": "player", "text": text}],
    })


def test_pipeline_marks_complaint_duplicate_for_human():
    reporter = uuid.uuid4().hex
    first, _ = triage(body(reporter, 0, "Arman_Petrov оскорбляет родню, 2.1"))
    assert not first.is_spam_or_duplicate
    second, meta = triage(body(reporter, 7, "Arman_Petrov оскорбляет родню 2.1 оп"), debug=True)
    assert second.is_spam_or_duplicate and second.needs_human
    assert second.category == first.category and second.priority == first.priority
    assert second.draft_reply.cited == ["ЖБ 1.2.9"] and meta["duplicate_of"]


def test_pipeline_report_duplicate_is_auto():
    reporter = uuid.uuid4().hex
    triage(body(reporter, 0, "меня убивают у спавна дм 45", kind="report"))
    second, _ = triage(body(reporter, 3, "меня убивают у спавна дм 45!!!", kind="report"))
    assert second.is_spam_or_duplicate and not second.needs_human
    assert second.draft_reply.cited == ["3.1"]


def test_no_reporter_no_dedup():
    req = TriageRequest.model_validate({"kind": "report", "messages": [{"role": "player", "text": "дм 45"}]})
    triage(req)
    again, meta = triage(req)
    assert meta["duplicate_of"] is None
