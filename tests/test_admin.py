"""Бонус: интерфейс админа — очередь, действия, проверка пунктов, выгрузка правок."""

import json

import pytest
from fastapi.testclient import TestClient

from copilot import config
from copilot.api import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_DB_PATH", tmp_path / "admin.db")
    with TestClient(app) as c:
        yield c


def add(client, text, kind="complaint", **extra):
    body = {"kind": kind, "messages": [{"role": "player", "text": text}], **extra}
    r = client.post("/api/admin/tickets", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_page_is_served(client):
    r = client.get("/admin")
    assert r.status_code == 200 and "Support Copilot" in r.text


def test_queue_sorted_by_priority_and_ticket_details(client):
    low = add(client, "+", kind="report")
    urgent = add(client, "взломали аккаунт, пропала машина", kind="report")
    insult = add(client, "Arman_Petrov оскорблял родню, 2.1 ОП", reported_nick="Arman_Petrov", images_count=1,
                 reference_answer="Игрок получит мут на 30 минут за 2.1 ОП.")
    items = client.get("/api/admin/queue").json()["items"]
    assert [i["id"] for i in items][0] == urgent["id"]
    assert items[-1]["id"] == low["id"]
    t = client.get(f"/api/admin/tickets/{insult['id']}").json()
    assert t["reference_answer"] and t["cited_rules"][0]["id"] == "2.1" and t["cited_rules"][0]["exists"]


def test_send_with_edit_and_category_fix_is_recorded(client):
    t = add(client, "проверьте логи, ничего не делал", reported_nick="Ivan_K", images_count=1)
    r = client.post(f"/api/admin/tickets/{t['id']}/action", json={
        "action": "send", "text": t["result"]["draft_reply"]["text"] + " Закрыто.",
        "category": "complaint.admin", "priority": "p2",
    })
    assert r.status_code == 200 and r.json()["status"] == "sent"
    fb = r.json()["feedback"][-1]
    assert fb["edited"] == 1 and fb["final_category"] == "complaint.admin" and fb["similarity"] < 1
    stats = client.get("/api/admin/stats").json()
    assert stats["queue"]["sent"] == 1 and stats["sent_edited"] == 1
    assert client.get("/api/admin/queue?status=new").json()["items"] == []


def test_escalation_needs_comment(client):
    t = add(client, "оплатил донат, не пришло", kind="report")
    assert client.post(f"/api/admin/tickets/{t['id']}/action", json={"action": "escalate"}).status_code == 422
    r = client.post(f"/api/admin/tickets/{t['id']}/action", json={"action": "escalate", "comment": "в платёжный отдел"})
    assert r.json()["status"] == "escalated"


def test_live_check_flags_nonexistent_rule(client):
    res = client.post("/api/admin/check", json={"text": "Нарушение п. 2.1 общих правил и п. 9.9 общих правил."}).json()
    assert [x["id"] for x in res["rules"]] == ["2.1"]
    assert any("9.9" in p for p in res["problems"])
    # регрессия: несуществующий пункт, за которым сразу идёт настоящий
    res = client.post("/api/admin/check", json={"text": "Нарушен п. 9.9 общих правил и п. 1.1 GZ."}).json()
    assert [x["id"] for x in res["rules"]] == ["GZ 1.1"] and any("9.9" in p for p in res["problems"])


def test_feedback_export_is_training_ready(client):
    t = add(client, "оск родни", reported_nick="Arman_Petrov", images_count=1)
    client.post(f"/api/admin/tickets/{t['id']}/action", json={"action": "send", "text": "Ок", "category": "complaint.insult"})
    lines = client.get("/api/admin/feedback/export").text.strip().splitlines()
    row = json.loads(lines[0])
    assert row["human_category"] == "complaint.insult" and row["kind"] == "complaint"
    assert all(m["role"] != "admin" for m in row["messages"])
