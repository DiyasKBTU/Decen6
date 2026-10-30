"""RAG: поиск находит нужный пункт, у каждого обращения есть основание, черновик цитирует только найденное."""

import pytest

from copilot.pipeline import triage
from copilot.retrieval import get_rule_retriever
from copilot.schemas import TriageRequest


@pytest.mark.parametrize("query, expected", [
    ("читер летает, софт аим", "5.2"),
    ("кинул на деньги, скам, взял машину и не отдал", "4.1"),
    ("убил в зз у банка", "GZ 1.1"),
    ("связал верёвкой и надел мешок в зз", "GZ 1.2"),
    ("расизм и оскорбление религии в чате", "2.4"),
    ("оск родни", "2.1"),
    ("продаёт аккаунт за реальные деньги", "4.4"),
])
def test_retrieval_finds_rule_in_top3(query, expected):
    ids = [h.id for h in get_rule_retriever().search(query, k=3)]
    assert expected in ids, ids


def test_core_sections_beat_event_rules():
    top = get_rule_retriever().search("дм в зз у банка", k=1)[0].id
    assert top.startswith("GZ"), top  # раньше первым был пункт правил семейных мероприятий


def run(text, kind="complaint", **extra):
    body = {"kind": kind, "messages": [{"role": "player", "text": text}], **extra}
    return triage(TriageRequest.model_validate(body), debug=True)


def test_every_ticket_gets_evidence_and_cites_only_found_rules():
    for text, kind in [("Arman_Petrov оск родни 2.1", "complaint"), ("софт аим, видео <url>", "complaint"),
                       ("убил в зз у банка, фрапс <url>", "complaint"), ("убил в зз у банка, скрины есть", "complaint"),
                       ("опру, за что бан", "complaint"), ("взял деньги за машину и не отдал, скам", "complaint"),
                       ("взломали аккаунт", "report"), ("как подать жалобу?", "report"), ("дайте тп", "report")]:
        resp, meta = run(text, kind, images_count=1)
        found = {r["id"] for r in meta["evidence"]["rules"]}
        assert found, text
        assert set(resp.draft_reply.cited) <= found, (text, resp.draft_reply.cited, found)
        assert all(r["cited"] == (r["id"] in resp.draft_reply.cited) for r in meta["evidence"]["rules"])


def test_rp_terms_come_from_rules_glossary():
    _, meta = run("дм и ск у спавна", images_count=1)
    assert {"DM", "SK"} <= set(meta["evidence"]["terms"])
    assert "убийство" in meta["evidence"]["terms"]["DM"].lower()
