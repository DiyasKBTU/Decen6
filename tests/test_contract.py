"""Контракт /triage и защитные правила. Запуск: pytest -q"""

from fastapi.testclient import TestClient

from copilot.api import app
from copilot.normalize import normalize
from copilot.policy import FORBIDDEN_PROMISES
from copilot.rules import get_index

client = TestClient(app)
CONTRACT_KEYS = {"lang", "category", "priority", "is_spam_or_duplicate", "extracted", "draft_reply",
                 "needs_human", "confidence", "reasons"}


def post(body: dict) -> dict:
    r = client.post("/triage", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert set(data) == CONTRACT_KEYS
    assert set(data["extracted"]) == {"nickname", "rule_refs", "has_evidence", "server"}
    assert set(data["draft_reply"]) == {"lang", "text", "cited"}
    for pid in data["draft_reply"]["cited"]:
        assert get_index().exists(pid), pid
    return data


def test_contract_example():
    data = post({"kind": "complaint", "messages": [{"role": "player", "attachments": 2,
                 "text": "Arman_Petrov оскорблял меня и мою семью в чате, скрины есть, 2.1 ОП"}]})
    assert data["category"] == "complaint.insult"
    assert data["extracted"] == {"nickname": "Arman_Petrov", "rule_refs": ["2.1"], "has_evidence": True, "server": None}
    assert data["draft_reply"]["cited"] == ["2.1"]
    assert data["needs_human"] is True  # в черновике наказание


def test_always_human_categories():
    pay = post({"kind": "report", "messages": [{"role": "player", "text": "оплатил гк, не пришло"}]})
    assert pay["category"] == "report.payment" and pay["needs_human"] and pay["priority"] == "p0"
    hack = post({"kind": "report", "messages": [{"role": "player", "text": "взломали аккаунт, пропала машина"}]})
    assert hack["category"] == "report.account" and hack["needs_human"]
    appeal = post({"kind": "complaint", "messages": [{"role": "player", "text": "опру, за что бан? я не оскорблял"}]})
    assert appeal["category"] == "complaint.admin" and appeal["needs_human"]


def test_offtopic_is_auto_and_spam():
    data = post({"kind": "report", "messages": [{"role": "player", "text": "ау"}]})
    assert data["category"] == "report.offtopic"
    assert data["is_spam_or_duplicate"] and not data["needs_human"]


def test_plus_is_event_teleport_request_not_spam():
    # «+» — ответ на объявление о телепорте на мероприятие; открыт ли телепорт, знает только админ
    data = post({"kind": "report", "messages": [{"role": "player", "text": "+"}]})
    assert data["category"] == "report.help"
    assert not data["is_spam_or_duplicate"] and data["needs_human"]


def test_admin_messages_are_ignored():
    body = {"kind": "report", "messages": [{"role": "player", "text": "помогите"},
                                           {"role": "admin", "text": "взлом аккаунта оплатил донат"}]}
    assert post(body)["category"] == "report.offtopic"


def test_kazakh_cp1251_restored_and_answered_in_kazakh():
    assert normalize("Машина аударылып ќалды")[0] == "Машина аударылып қалды"
    data = post({"kind": "report", "server_lang": "kk", "messages": [{"role": "player", "text": "Машина аударылып ќалды"}]})
    assert data["lang"] == "kk" and data["draft_reply"]["lang"] == "kk"


def test_ukrainian_is_not_kazakh():
    data = post({"kind": "complaint", "messages": [{"role": "player", "text": "2.1оп дати бан щоб знав як потрібно розмовляти"}]})
    assert data["lang"] == "ru"


def test_uzbek_draft():
    data = post({"kind": "complaint", "server_lang": "uz", "reported_nick": "Lev_Walker55", "images_count": 2,
                 "messages": [{"role": "player", "text": "onadan sukindi va mani uldirdi ban berizla"}]})
    assert data["lang"] == "uz" and data["draft_reply"]["lang"] == "uz" and "2.1" in data["draft_reply"]["cited"]


def test_scam_draft_does_not_promise_refund():
    with_video = post({"kind": "complaint", "reported_nick": "Ivan_K", "images_count": 1,
                       "messages": [{"role": "player", "text": "взял деньги за машину и не отдал, скам, фрапс <url>"}]})
    assert {"4.1", "1.4"} <= set(with_video["draft_reply"]["cited"])  # наказание + «имущество не восстанавливается»
    assert with_video["needs_human"]


def test_dm_cheat_scam_without_video_ask_for_video():
    # train: по таким жалобам только со скриншотами админы в ~60% просят видео и лишь в ~10% наказывают
    for text in ["дм у банка без причины", "читер с аимом, софт", "взял деньги за машину и не отдал, скам"]:
        data = post({"kind": "complaint", "reported_nick": "Ivan_K", "images_count": 2,
                     "messages": [{"role": "player", "text": text}]})
        assert {"ЖБ 1.2.2", "ЖБ 1.2.1"} <= set(data["draft_reply"]["cited"]), text
        assert "видео" in data["draft_reply"]["text"] and data["needs_human"]
    insult = post({"kind": "complaint", "reported_nick": "Ivan_K", "images_count": 1,
                   "messages": [{"role": "player", "text": "оск родни"}]})
    assert "2.1" in insult["draft_reply"]["cited"]  # оскорбление по скриншоту видно — предлагаем наказание


def test_promises_always_need_human():
    for text in ["оск родни", "дм в зз у банка", "софт аим", "+", "застрял в текстурах", "как подать жалобу?"]:
        for kind in ("complaint", "report"):
            data = post({"kind": kind, "messages": [{"role": "player", "text": text}]})
            if FORBIDDEN_PROMISES.search(data["draft_reply"]["text"]):
                assert data["needs_human"], (kind, text)


def test_rule_refs_parsing():
    ix = get_index()
    assert ix.parse_refs("2.1оп и 1.1 gz, п. 1.2.7 правил подачи жалоб") == ["2.1", "GZ 1.1", "ЖБ 1.2.7"]
    assert ix.parse_refs("3,5,7") == [] and ix.parse_refs("170") == []
