"""Этап 4 без сети: фейковый клиент вместо API. Проверяем запрос, проверки и откат к шаблону."""

import json
import re
from types import SimpleNamespace

import pytest

from copilot import llm
from copilot.pipeline import triage
from copilot.schemas import TriageRequest


class FakeClient:
    """Отвечает payload(kwargs) -> dict; запоминает последний запрос."""

    def __init__(self, payload, usage=(1200, 150)):
        self.payload = payload
        self.usage = usage
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        data = self.payload(kwargs)
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=json.dumps(data, ensure_ascii=False))],
            usage=SimpleNamespace(input_tokens=self.usage[0], output_tokens=self.usage[1],
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )


def answer(text, cited=(), hint=False, reason=""):
    return lambda kwargs: {"text": text, "cited": list(cited), "needs_human_hint": hint, "hint_reason": reason}


@pytest.fixture
def fake():
    def install(payload):
        client = FakeClient(payload)
        llm.set_drafter(llm.Drafter(client=client))
        return client
    yield install
    llm.set_drafter(None)


def run(body):
    resp, meta = triage(TriageRequest.model_validate(body), debug=True)
    return resp.model_dump(), meta


INSULT = {"kind": "complaint", "reported_nick": "Arman_Petrov", "images_count": 2,
          "messages": [{"role": "player", "text": "Arman_Petrov оскорблял мою семью, 2.1 ОП"}]}


def test_llm_draft_is_used_and_request_is_well_formed(fake):
    client = fake(answer("Здравствуйте! По предоставленным материалам игрок Arman_Petrov нарушил п. 2.1 общих правил "
                         "(оскорбление родных). Игрок Arman_Petrov получит наказание: мут на 30 минут.", ["2.1"]))
    data, meta = run(INSULT)
    assert meta["draft_source"] == "llm"
    assert data["draft_reply"]["cited"] == ["2.1"] and data["needs_human"]
    req = client.calls[0]
    assert req["model"] == "claude-haiku-5-5"
    assert "temperature" not in req and "top_p" not in req
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    schema = req["output_config"]["format"]["schema"]
    assert "2.1" in schema["properties"]["cited"]["items"]["enum"]
    assert "<ticket>" in req["messages"][0]["content"]
    assert meta["llm_cost_usd"] == pytest.approx((1200 * 0.10 + 150 * 0.50) / 1e6)


def test_invented_reference_falls_back_to_template(fake):
    fake(answer("Здравствуйте! Нарушен п. 9.9 общих правил.", []))
    data, meta = run(INSULT)
    assert meta["draft_source"] == "template" and meta["llm_fallback_reason"] == "rejected:bad_reference"
    assert data["draft_reply"]["cited"] == ["2.1"]


def test_real_but_not_allowed_reference_rejected(fake):
    fake(answer("Здравствуйте! Нарушен п. 4.7 общих правил.", []))
    _, meta = run(INSULT)
    assert meta["llm_fallback_reason"] == "rejected:reference_not_allowed"


def test_unban_promise_rejected(fake):
    fake(answer("Здравствуйте! Мы снимем бан в ближайшее время.", []))
    _, meta = run({"kind": "complaint", "messages": [{"role": "player", "text": "опру, за что бан? я не оскорблял"}]})
    assert meta["llm_fallback_reason"] == "rejected:forbidden_promise"


def test_punishment_outside_decision_rejected(fake):
    fake(answer("Здравствуйте! Игрок получит бан, сообщение передано администратору.", []))
    _, meta = run({"kind": "report", "messages": [{"role": "player", "text": "меня дмят у спавна 45 дм"}]})
    assert meta["llm_fallback_reason"] == "rejected:punishment_outside_decision"


def test_wrong_language_rejected(fake):
    fake(answer("Здравствуйте! Ваш запрос передан администратору, ожидайте помощи.", []))
    data, meta = run({"kind": "report", "server_lang": "kk",
                      "messages": [{"role": "player", "text": "мен текстураға тұрып қалдым көмектесіңіз"}]})
    assert meta["llm_fallback_reason"].startswith("rejected:wrong_language")
    assert data["draft_reply"]["lang"] == "kk"


def test_unknown_nickname_rejected(fake):
    fake(answer("Здравствуйте, Dias_Volkov! Жалоба на игрока Arman_Petrov рассмотрена, нарушен п. 2.1 общих правил.", ["2.1"]))
    _, meta = run(INSULT)
    assert meta["llm_fallback_reason"] == "rejected:unknown_nickname"


def test_llm_error_falls_back(fake):
    class Failing(llm.Drafter):
        def draft(self, ctx):
            raise llm.LLMError("timeout")

    llm.set_drafter(Failing(client=object()))
    data, meta = run(INSULT)
    assert meta["draft_source"] == "template" and meta["llm_fallback_reason"] == "llm_error:timeout"
    assert data["draft_reply"]["cited"] == ["2.1"]


def test_model_hint_forces_human(fake):
    fake(answer("Здравствуйте! Ваш вопрос передан администратору, ожидайте ответа.", [], True, "непонятно, о чём вопрос"))
    data, meta = run({"kind": "report", "messages": [{"role": "player", "text": "как устроиться в полицию?"}]})
    assert meta["draft_source"] == "llm" and data["needs_human"]
    assert any("модель просит проверить" in r for r in data["reasons"])


def test_simple_situations_skip_llm(fake):
    client = fake(answer("не должен вызываться"))
    _, meta = run({"kind": "report", "messages": [{"role": "player", "text": "+"}]})
    assert meta["llm_fallback_reason"] == "template_is_enough" and not client.calls


def test_validator_accepts_own_templates_in_all_languages(fake):
    """Если модель вернёт текст шаблона, проверки не должны его отвергать."""
    def echo(kwargs):
        content = kwargs["messages"][0]["content"]
        text = re.search(r"Черновик-шаблон \(можно взять за основу и улучшить\): (.*)", content).group(1)
        ids = re.findall(r'id "([^"]+)"', content)
        return {"text": text, "cited": [i for i in ids if i.split()[-1] in text], "needs_human_hint": False, "hint_reason": ""}

    fake(echo)
    cases = [
        ("complaint", None, "Arman_Petrov оскорблял мою семью, 2.1 ОП"),
        ("complaint", "uz", "onadan sukindi va mani uldirdi ban berizla"),
        ("complaint", "kk", "Мен оны қорлады деп шағым жазамын, ол маған балағат сөз айтты"),
        ("complaint", None, "опру, за что бан? я не оскорблял"),
        ("report", None, "взломали аккаунт, пропала машина"),
        ("report", "uz", "donat qildim pul tushmadi to‘lov"),
        ("report", None, "he killed me for no reason dm"),
    ]
    for kind, server_lang, text in cases:
        data, meta = run({"kind": kind, "server_lang": server_lang, "reported_nick": "Arman_Petrov" if kind == "complaint" else None,
                          "messages": [{"role": "player", "text": text}]})
        assert meta["draft_source"] == "llm", (text, meta["llm_fallback_reason"])
