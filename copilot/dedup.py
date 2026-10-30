"""Бонус: дубли — «1 игрок, 1 проблема, 5 обращений за час».

Калибровка на train (scripts/ и docs/research-log.md): админы помечают повтор дубликатом почти
только тогда, когда первое обращение ещё без ответа. После ответа повтор — это обычно
новая подача (с доказательствами, по новому эпизоду), и склеивать её нельзя.

Правила:
- жалоба: тот же заявитель и тот же обвиняемый, предыдущая за 24 ч, ещё без ответа,
  тексты похожи (косинус ≥ 0.6, а в пределах часа ≥ 0.4);
- репорт: тот же игрок, предыдущий за 30 мин, ещё без ответа, похожий текст (≥ 0.5),
  либо тот же текст за 10 мин (флуд).

История: прошлые обращения из train (artifacts/dedup_history.pkl) + всё, что прошло
через сервис. Учитываются только обращения раньше текущего по времени.
"""

from __future__ import annotations

import datetime as dt
import logging
import pickle
import threading
from bisect import insort
from dataclasses import dataclass, field
from functools import lru_cache

from sklearn.feature_extraction.text import HashingVectorizer

from . import config
from .normalize import normalize

log = logging.getLogger(__name__)

COMPLAINT_WINDOW_S = 24 * 3600
REPORT_WINDOW_S = 30 * 60
FLOOD_WINDOW_S = 10 * 60
UNANSWERED_ASSUMED_S = 48 * 3600  # срок рассмотрения жалобы: без данных об ответе считаем, что ещё ждёт
# На train: первая без ответа + сходство ≥ 0.6 -> вердикт «дубликат» у 26%, «одобрено» у 11%;
# сходство 0.4–0.6 в пределах часа -> 18% и 8% (то же соотношение, полнота в 1,5 раза выше).
# Если на первую уже ответили, даже одинаковый текст — «дубликат» 14% против 14% «одобрено»: не склеиваем.
COMPLAINT_MIN_SIM = 0.6
COMPLAINT_MIN_SIM_1H = 0.4
REPORT_MIN_SIM = 0.5

_vec = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=2**18, alternate_sign=False, norm="l2")


def parse_ts(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


@dataclass(order=True)
class Record:
    created: float
    ticket_id: str = field(compare=False)
    kind: str = field(compare=False)
    target: str | None = field(compare=False)
    text: str = field(compare=False)
    answered: float | None = field(compare=False, default=None)


@dataclass
class Duplicate:
    of_ticket: str
    minutes_ago: int
    similarity: float
    kind: str


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 1.0 if a == b else 0.0
    m = _vec.transform([a, b])
    return float(m[0].multiply(m[1]).sum())


def record_from_row(row: dict) -> Record | None:
    created = parse_ts(row.get("created_at"))
    if created is None or not row.get("reporter_hash"):
        return None
    msgs = row.get("messages", [])
    player = " ".join(m.get("text", "") for m in msgs if m.get("role") == "player")
    answered = next((parse_ts(m.get("ts")) for m in msgs if m.get("role") == "admin" and m.get("ts")), None)
    return Record(
        created=created,
        ticket_id=str(row.get("ticket_id") or ""),
        kind=row["kind"],
        target=row.get("reported_hash") or row.get("reported_nick"),
        text=normalize(player)[0].lower()[:400],
        answered=answered,
    )


class DedupStore:
    def __init__(self, history: dict[str, list[Record]] | None = None):
        self._by_reporter: dict[str, list[Record]] = history or {}
        self._lock = threading.Lock()

    def check_and_add(self, reporter: str, rec: Record) -> Duplicate | None:
        with self._lock:
            past = self._by_reporter.setdefault(reporter, [])
            dup = self._find(past, rec)
            if not any(p.ticket_id == rec.ticket_id and rec.ticket_id for p in past):
                insort(past, rec)
            return dup

    def _find(self, past: list[Record], rec: Record) -> Duplicate | None:
        best: Duplicate | None = None
        for p in reversed(past):
            gap = rec.created - p.created
            if gap < 0 or p.ticket_id == rec.ticket_id and rec.ticket_id:
                continue
            window = COMPLAINT_WINDOW_S if rec.kind == "complaint" else REPORT_WINDOW_S
            if gap > window:
                break
            if p.kind != rec.kind:
                continue
            pending = (p.answered is None and gap <= UNANSWERED_ASSUMED_S) or (p.answered is not None and p.answered > rec.created)
            sim = similarity(p.text, rec.text)
            if rec.kind == "complaint":
                similar = sim >= COMPLAINT_MIN_SIM or (sim >= COMPLAINT_MIN_SIM_1H and gap <= 3600)
                ok = pending and rec.target and p.target == rec.target and similar
            else:
                ok = (pending and sim >= REPORT_MIN_SIM) or (gap <= FLOOD_WINDOW_S and p.text == rec.text)
            if ok and (best is None or sim > best.similarity):
                best = Duplicate(p.ticket_id, int(gap // 60), round(sim, 2), rec.kind)
        return best


def enabled() -> bool:
    return config.DEDUP_MODE not in ("0", "off", "false")


@lru_cache(maxsize=1)
def get_store() -> DedupStore:
    history: dict[str, list[Record]] = {}
    path = config.DEDUP_HISTORY_PATH
    if path.exists():
        with open(path, "rb") as f:
            history = pickle.load(f)
    else:
        log.warning("история обращений для дублей не найдена (%s): дубли только среди новых; make artifacts", path)
    return DedupStore(history)


def check_and_add(row: dict) -> Duplicate | None:
    rec = record_from_row(row)
    if rec is None:
        return None  # без reporter_hash и created_at дубль не определить
    rec.answered = None  # ответ на новое обращение сервису неизвестен
    return get_store().check_and_add(row["reporter_hash"], rec)
