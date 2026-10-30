"""Бонус: API интерфейса админа (/api/admin/...) и страница /admin."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel

from . import config, template_library
from .admin_store import get_store
from .pipeline import triage
from .policy import HARD_FORBIDDEN, PUNISHMENT_PROMISE, check_citations
from .rules import get_index
from .schemas import TriageRequest
from .taxonomy import get_taxonomy

STATIC = Path(__file__).parent / "static"
router = APIRouter()


class NewTicket(TriageRequest):
    reference_answer: str | None = None  # ответ админа из выгрузки — только для демо


class Action(BaseModel):
    action: Literal["send", "escalate"]
    text: str = ""
    category: str | None = None
    priority: str | None = None
    comment: str | None = None
    admin: str | None = None


class TextCheck(BaseModel):
    text: str


def _rule_info(pid: str) -> dict:
    p = get_index().get(pid)
    return {"id": pid, "exists": p is not None, "section": p.section_name if p else None,
            "text": p.text if p else None, "short": p.short if p else None}


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/admin")


@router.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    return FileResponse(STATIC / "admin.html")


@router.get("/api/admin/meta")
def meta() -> dict:
    tax = get_taxonomy()
    return {
        "categories": [{"code": c, "title": v["title"], "kind": v["kind"]} for c, v in tax.categories.items()],
        "priorities": tax.priorities,
    }


@router.get("/api/admin/queue")
def queue(status: str = "new", category: str | None = None, needs_human: bool | None = None,
          lang: str | None = None, q: str | None = None, limit: int = Query(300, le=2000)) -> dict:
    store = get_store()
    return {"items": store.queue(status, category, needs_human, lang, q, limit), "stats": store.stats()}


@router.post("/api/admin/tickets")
def add_ticket(ticket: NewTicket) -> dict:
    """Новое обращение (например, от игры): триаж и постановка в очередь."""
    req = TriageRequest.model_validate(ticket.model_dump(exclude={"reference_answer"}))
    response, meta_ = triage(req)
    request = req.model_dump()
    request["messages"] = [m for m in request["messages"] if m["role"] != "admin"]
    tid = get_store().add(request, response.model_dump(), meta_, ticket.reference_answer)
    return get_store().get(tid)


@router.get("/api/admin/tickets/{tid}")
def get_ticket(tid: str) -> dict:
    ticket = get_store().get(tid)
    if ticket is None:
        raise HTTPException(404, "обращение не найдено")
    ticket["cited_rules"] = [_rule_info(pid) for pid in ticket["result"]["draft_reply"]["cited"]]
    return ticket


@router.post("/api/admin/tickets/{tid}/action")
def act(tid: str, body: Action) -> dict:
    if body.action == "escalate" and not (body.comment or "").strip():
        raise HTTPException(422, "для эскалации нужен комментарий")
    try:
        return get_store().act(tid, body.action, body.text, body.category, body.priority, body.comment, body.admin)
    except KeyError:
        raise HTTPException(404, "обращение не найдено") from None


@router.post("/api/admin/check")
def check_text(body: TextCheck) -> dict:
    """Пункты правил в тексте админа: существуют ли они."""
    cited, problems = check_citations(body.text, [], get_index())
    return {"rules": [_rule_info(pid) for pid in cited], "problems": problems}


@router.get("/api/admin/rules/{rule_id}")
def get_rule(rule_id: str) -> dict:
    info = _rule_info(rule_id)
    if not info["exists"]:
        raise HTTPException(404, "такого пункта нет")
    return info


@router.get("/api/admin/stats")
def stats() -> dict:
    return get_store().stats()


class TemplateDecision(BaseModel):
    status: Literal["accepted", "rejected"]
    text: str = ""
    admin: str | None = None


GREETINGS = {"ru": "Здравствуйте!", "uz": "Assalomu alaykum!", "kk": "Сәлеметсіз бе!", "en": "Hello!"}


def proposed_text(s: dict) -> str:
    """Типичный ответ админов -> текст шаблона в нашем стиле: приветствие, без ников, с точкой."""
    body = re.sub(r"\s*<ник>\s*,?", " ", s["admin_answer"]).strip(" ,.")
    body = re.sub(r"\s+", " ", body)
    body = body[:1].upper() + body[1:]
    if body and body[-1] not in ".!?":
        body += "."
    return f"{GREETINGS.get(s['lang'], '')} {body}".strip()


def _suggestions() -> list[dict]:
    path = config.TEMPLATE_SUGGESTIONS_PATH
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


@router.get("/api/admin/templates")
def list_templates() -> dict:
    decisions = get_store().template_decisions()
    items = []
    for s in _suggestions():
        d = decisions.get(s["cluster"])
        items.append({**s, "proposed_text": proposed_text(s), "status": d["status"] if d else "new",
                      "accepted_text": d["text"] if d else None, "decided_at": d["decided_at"] if d else None})
    return {"items": items, "accepted": sum(i["status"] == "accepted" for i in items)}


@router.post("/api/admin/templates/{cluster}/decision")
def decide_template(cluster: str, body: TemplateDecision) -> dict:
    s = next((x for x in _suggestions() if x["cluster"] == cluster), None)
    if s is None:
        raise HTTPException(404, "предложение не найдено")
    text = body.text.strip() or proposed_text(s)
    if body.status == "accepted":
        _, problems = check_citations(text, [], get_index())
        if problems:
            raise HTTPException(422, "; ".join(problems))
        if HARD_FORBIDDEN.search(text) or PUNISHMENT_PROMISE.search(text):
            raise HTTPException(422, "шаблон отправляется автоматически: в нём не должно быть обещаний разбана, "
                                     "возврата или наказания другого игрока")
    get_store().decide_template(cluster, s.get("id", ""), s["kind"], s["lang"], text, body.status, body.admin)
    template_library.accepted.reload()
    return {"cluster": cluster, "status": body.status, "text": text}


@router.get("/api/admin/feedback/export", response_class=PlainTextResponse)
def export_feedback() -> str:
    """JSONL: обращение + решение человека. Вход для scripts/train_classifier.py --feedback."""
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in get_store().export_feedback())
