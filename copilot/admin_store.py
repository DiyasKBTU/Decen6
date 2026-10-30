"""Бонус: очередь админа и журнал его действий (SQLite).

Журнал feedback — обучающий сигнал: исправленные категория и приоритет становятся
человеческой разметкой для классификатора (scripts/train_classifier.py --feedback),
а отправленные тексты — образцами ответов. База лежит в out/ (тексты обращений — вне git).
"""

from __future__ import annotations

import datetime as dt
import difflib
import json
import sqlite3
import threading
import uuid
from pathlib import Path

from . import config

PRIORITY_ORDER = {"p0": 0, "p1": 1, "p2": 2, "p3": 3}
SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    created_at TEXT,
    kind TEXT,
    request_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    meta_json TEXT NOT NULL,
    reference_answer TEXT,      -- ответ админа из выгрузки: только для демо и сравнения
    priority TEXT,
    category TEXT,
    lang TEXT,
    needs_human INTEGER,
    is_duplicate INTEGER,
    status TEXT NOT NULL DEFAULT 'new',   -- new | sent | escalated
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS tickets_queue ON tickets(status, priority, created_at);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    action TEXT NOT NULL,       -- send | escalate
    admin TEXT,
    draft_text TEXT,
    final_text TEXT,
    edited INTEGER,
    similarity REAL,            -- насколько итог похож на черновик (1.0 — без правок)
    model_category TEXT,
    final_category TEXT,
    model_priority TEXT,
    final_priority TEXT,
    draft_source TEXT,
    comment TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS templates (
    cluster TEXT PRIMARY KEY,   -- kind:lang:номер кластера
    suggestion_id TEXT,
    kind TEXT,
    lang TEXT,
    text TEXT,
    status TEXT NOT NULL,       -- accepted | rejected
    admin TEXT,
    decided_at TEXT
);
"""


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


class AdminStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # --- очередь ---
    def add(self, request: dict, result: dict, meta: dict, reference_answer: str | None = None) -> str:
        tid = request.get("ticket_id") or uuid.uuid4().hex[:16]
        request = {**request, "ticket_id": tid}
        with self._lock, self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO tickets (id, created_at, kind, request_json, result_json, meta_json,
                   reference_answer, priority, category, lang, needs_human, is_duplicate, status, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', ?)""",
                (tid, request.get("created_at") or _now(), request["kind"], json.dumps(request, ensure_ascii=False),
                 json.dumps(result, ensure_ascii=False), json.dumps(meta, ensure_ascii=False, default=str),
                 reference_answer, result["priority"], result["category"], result["lang"], int(result["needs_human"]),
                 int(result["is_spam_or_duplicate"] and bool(meta.get("duplicate_of"))), _now()),
            )
        return tid

    def queue(self, status: str = "new", category: str | None = None, needs_human: bool | None = None,
              lang: str | None = None, q: str | None = None, limit: int = 300) -> list[dict]:
        sql, args = "SELECT * FROM tickets WHERE 1=1", []
        if status == "done":
            sql += " AND status IN ('sent', 'escalated')"
        elif status and status != "all":
            sql += " AND status = ?"
            args.append(status)
        if category:
            sql += " AND category = ?"
            args.append(category)
        if needs_human is not None:
            sql += " AND needs_human = ?"
            args.append(int(needs_human))
        if lang:
            sql += " AND lang = ?"
            args.append(lang)
        if q:
            sql += " AND request_json LIKE ?"
            args.append(f"%{q}%")
        order = "updated_at DESC" if status == "done" else (
            "CASE priority WHEN 'p0' THEN 0 WHEN 'p1' THEN 1 WHEN 'p2' THEN 2 ELSE 3 END, created_at")
        sql += f" ORDER BY {order} LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            rows = c.execute(sql, args).fetchall()
        return [self._summary(r) for r in rows]

    @staticmethod
    def _summary(r: sqlite3.Row) -> dict:
        req = json.loads(r["request_json"])
        player = " ".join(m.get("text", "") for m in req.get("messages", []) if m.get("role") == "player")
        return {
            "id": r["id"], "created_at": r["created_at"], "kind": r["kind"], "priority": r["priority"],
            "category": r["category"], "lang": r["lang"], "needs_human": bool(r["needs_human"]),
            "is_duplicate": bool(r["is_duplicate"]), "status": r["status"], "snippet": player[:140],
        }

    def get(self, tid: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM tickets WHERE id = ?", (tid,)).fetchone()
            if not r:
                return None
            fb = c.execute("SELECT * FROM feedback WHERE ticket_id = ? ORDER BY id", (tid,)).fetchall()
        return {
            **self._summary(r),
            "request": json.loads(r["request_json"]),
            "result": json.loads(r["result_json"]),
            "meta": json.loads(r["meta_json"]),
            "reference_answer": r["reference_answer"],
            "feedback": [dict(x) for x in fb],
        }

    # --- действия админа ---
    def act(self, tid: str, action: str, text: str, category: str | None, priority: str | None,
            comment: str | None, admin: str | None) -> dict:
        ticket = self.get(tid)
        if ticket is None:
            raise KeyError(tid)
        result, meta = ticket["result"], ticket["meta"]
        draft = result["draft_reply"]["text"]
        similarity = difflib.SequenceMatcher(None, draft, text or "").ratio()
        status = {"send": "sent", "escalate": "escalated"}[action]
        with self._lock, self._conn() as c:
            c.execute(
                """INSERT INTO feedback (ticket_id, action, admin, draft_text, final_text, edited, similarity,
                   model_category, final_category, model_priority, final_priority, draft_source, comment, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (tid, action, admin, draft, text, int((text or "").strip() != draft.strip()), round(similarity, 3),
                 result["category"], category or result["category"], result["priority"], priority or result["priority"],
                 meta.get("draft_source"), comment, _now()),
            )
            c.execute("UPDATE tickets SET status = ?, category = ?, priority = ?, updated_at = ? WHERE id = ?",
                      (status, category or result["category"], priority or result["priority"], _now(), tid))
        return self.get(tid)

    def stats(self) -> dict:
        with self._conn() as c:
            by_status = {r[0]: r[1] for r in c.execute("SELECT status, COUNT(*) FROM tickets GROUP BY status")}
            fb = c.execute(
                """SELECT COUNT(*), SUM(action='send' AND edited=0), SUM(action='send' AND edited=1), SUM(action='escalate'),
                   SUM(final_category != model_category), SUM(final_priority != model_priority), AVG(similarity)
                   FROM feedback""").fetchone()
        n = fb[0] or 0
        return {
            "queue": {"new": by_status.get("new", 0), "sent": by_status.get("sent", 0), "escalated": by_status.get("escalated", 0)},
            "actions": n,
            "sent_as_is": fb[1] or 0,
            "sent_edited": fb[2] or 0,
            "escalated": fb[3] or 0,
            "category_fixed": fb[4] or 0,
            "priority_fixed": fb[5] or 0,
            "avg_similarity": round(fb[6], 3) if fb[6] is not None else None,
        }

    # --- библиотека шаблонов ---
    def decide_template(self, cluster: str, suggestion_id: str, kind: str, lang: str, text: str, status: str,
                        admin: str | None) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO templates (cluster, suggestion_id, kind, lang, text, status, admin, decided_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (cluster, suggestion_id, kind, lang, text, status, admin, _now()),
            )

    def template_decisions(self) -> dict[str, dict]:
        with self._conn() as c:
            return {r["cluster"]: dict(r) for r in c.execute("SELECT * FROM templates")}

    def accepted_templates(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM templates WHERE status = 'accepted'")]

    def export_feedback(self) -> list[dict]:
        """Обращение + решение человека — строки для обучения (формат как у выгрузки)."""
        with self._conn() as c:
            rows = c.execute(
                """SELECT f.*, t.request_json FROM feedback f JOIN tickets t ON t.id = f.ticket_id
                   WHERE f.id IN (SELECT MAX(id) FROM feedback GROUP BY ticket_id) ORDER BY f.id""").fetchall()
        out = []
        for r in rows:
            req = json.loads(r["request_json"])
            out.append({
                **{k: v for k, v in req.items() if k != "messages"},
                "messages": [m for m in req.get("messages", []) if m.get("role") != "admin"],
                "human_category": r["final_category"], "human_priority": r["final_priority"],
                "model_category": r["model_category"], "model_priority": r["model_priority"],
                "action": r["action"], "draft_text": r["draft_text"], "final_text": r["final_text"],
                "edited": bool(r["edited"]), "similarity": r["similarity"], "draft_source": r["draft_source"],
                "comment": r["comment"], "decided_at": r["created_at"],
            })
        return out


_store: AdminStore | None = None


def get_store() -> AdminStore:
    global _store
    if _store is None or _store.path != config.ADMIN_DB_PATH:
        _store = AdminStore(config.ADMIN_DB_PATH)
    return _store
