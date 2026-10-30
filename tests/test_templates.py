"""Бонус: новые шаблоны — принятие во вкладке «Шаблоны» и применение к похожим обращениям."""

import json
import pickle

import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer

from copilot import config, template_library
from copilot.api import app
from copilot.template_library import cluster_text

THANKS = ["спасибо большое", "спасибо админ", "благодарю за помощь", "спасибо вам огромное", "спасибо за помощь"]
OTHER = ["меня убили без причины", "застрял в текстурах помогите", "как устроиться в полицию", "дм у банка 45"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    # маленькая модель кластеров: 0 — благодарности, 1 — всё остальное
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit([cluster_text(t) for t in THANKS + OTHER])
    centroids = [vec.transform([cluster_text(t) for t in group]).mean(axis=0).A1 for group in (THANKS, OTHER)]
    clusters = tmp_path / "clusters.pkl"
    clusters.write_bytes(pickle.dumps({"report:ru": {"vectorizer": vec, "centroids": __import__("numpy").vstack(centroids),
                                                       "kind": "report", "lang": "ru"}}))
    suggestions = tmp_path / "suggestions.json"
    suggestions.write_text(json.dumps([{
        "id": "T01", "cluster": "report:ru:0", "kind": "report", "lang": "ru", "size": 106, "answer_share": 0.71,
        "similarity_to_ours": 0.0, "score": 30, "admin_answer": "приятной игры на Grand Mobile <ник>", "refs": [],
        "terms": ["спасибо"], "examples": THANKS[:3], "our_situations": [["report_offtopic", 90]],
        "admin_verdicts": [], "top_situation": "report_offtopic",
    }], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(config, "ADMIN_DB_PATH", tmp_path / "admin.db")
    monkeypatch.setattr(config, "TEMPLATE_CLUSTERS_PATH", clusters)
    monkeypatch.setattr(config, "TEMPLATE_SUGGESTIONS_PATH", suggestions)
    template_library.get_clusters.cache_clear()
    template_library.accepted.reload()
    with TestClient(app) as c:
        yield c
    template_library.get_clusters.cache_clear()
    template_library.accepted._by_cluster = None


def triage(client, text):
    return client.post("/triage?debug=true", json={"kind": "report", "messages": [{"role": "player", "text": text}]}).json()


def test_suggestion_listed_with_clean_proposed_text(client):
    item = client.get("/api/admin/templates").json()["items"][0]
    assert item["status"] == "new"
    assert item["proposed_text"] == "Здравствуйте! Приятной игры на Grand Mobile."  # без «<ник>»


def test_accept_rejects_bad_references_and_promises(client):
    url = "/api/admin/templates/report:ru:0/decision"
    assert client.post(url, json={"status": "accepted", "text": "Спасибо! См. п. 9.9 общих правил."}).status_code == 422
    assert client.post(url, json={"status": "accepted", "text": "Спасибо! Мы снимем бан."}).status_code == 422


def test_accepted_template_is_applied_to_similar_tickets_only(client):
    before = triage(client, "спасибо большое за помощь")
    assert before["debug"]["draft_source"] != "library"
    r = client.post("/api/admin/templates/report:ru:0/decision", json={"status": "accepted", "text": ""})
    assert r.status_code == 200
    after = triage(client, "спасибо большое за помощь")
    assert after["debug"]["draft_source"] == "library"
    assert after["draft_reply"]["text"] == "Здравствуйте! Приятной игры на Grand Mobile."
    assert not after["needs_human"]
    unrelated = triage(client, "застрял в текстурах помогите")
    assert unrelated["debug"]["draft_source"] != "library"
    payment = triage(client, "спасибо, но донат не пришёл, оплатил гк")  # решения правил шаблон не трогает
    assert payment["category"] == "report.payment" and payment["debug"]["draft_source"] != "library"


def test_rejected_template_is_not_applied(client):
    client.post("/api/admin/templates/report:ru:0/decision", json={"status": "rejected"})
    assert triage(client, "спасибо большое")["debug"]["draft_source"] != "library"
    assert client.get("/api/admin/templates").json()["items"][0]["status"] == "rejected"
