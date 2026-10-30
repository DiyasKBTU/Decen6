"""Бонус: библиотека шаблонов из кластеров похожих обращений.

scripts/suggest_templates.py находит кластеры, где админы отвечают единообразно, а сервис —
иначе, и предлагает шаблон. Админ принимает или правит его во вкладке «Шаблоны» (/admin).
Принятый шаблон сервис применяет к новым обращениям, близким к центру того же кластера.
"""

from __future__ import annotations

import logging
import pickle
import re
import threading
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import config
from .normalize import normalize

log = logging.getLogger(__name__)

_NICK = re.compile(r"\b[A-Z][a-z]{1,15}_[A-Z][a-z]{1,15}\d{0,3}\b")
# Насколько обращение должно быть близко к центру кластера, чтобы применить его шаблон
MIN_CENTROID_SIM = 0.35
# Принятый шаблон заменяет только общие ответы; решения правил (платёж, наказание, апелляция) не трогает
REPLACEABLE_SITUATIONS = {"report_question", "report_help", "report_complaint", "report_offtopic", "complaint_review"}


def cluster_text(player_text: str, rule_refs_raw: str | None = None) -> str:
    text = _NICK.sub(" ник ", normalize(player_text or "")[0]).lower()
    text = re.sub(r"\d", "0", text)
    if rule_refs_raw:
        text += " | " + str(rule_refs_raw).lower()
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Match:
    cluster: str
    similarity: float


@lru_cache(maxsize=1)
def get_clusters() -> dict | None:
    path = config.TEMPLATE_CLUSTERS_PATH
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def nearest_cluster(kind: str, lang: str, player_text: str, rule_refs_raw: str | None) -> Match | None:
    models = get_clusters()
    model = models.get(f"{kind}:{lang}") if models else None
    if model is None:
        return None
    x = model["vectorizer"].transform([cluster_text(player_text, rule_refs_raw)])
    centroids = model["centroids"]
    sims = np.asarray(x @ centroids.T).ravel() / (np.linalg.norm(centroids, axis=1) + 1e-9)
    c = int(sims.argmax())
    return Match(f"{kind}:{lang}:{c}", float(sims[c]))


class AcceptedTemplates:
    """Принятые админами шаблоны (из admin.db); кэш обновляется при принятии."""

    def __init__(self):
        self._lock = threading.Lock()
        self._by_cluster: dict[str, dict] | None = None

    def reload(self) -> None:
        from .admin_store import get_store

        with self._lock:
            self._by_cluster = {t["cluster"]: t for t in get_store().accepted_templates()}

    def _ensure(self) -> dict[str, dict]:
        if self._by_cluster is None:
            try:
                self.reload()
            except Exception as exc:  # база недоступна — работаем без библиотеки
                log.warning("библиотека шаблонов недоступна: %s", exc)
                self._by_cluster = {}
        return self._by_cluster

    def get(self, cluster: str) -> dict | None:
        return self._ensure().get(cluster)

    def has_any(self, kind: str, lang: str) -> bool:
        prefix = f"{kind}:{lang}:"
        return any(c.startswith(prefix) for c in self._ensure())


accepted = AcceptedTemplates()


def library_draft(kind: str, lang: str, situation: str, player_text: str, rule_refs_raw: str | None) -> tuple[dict, Match] | None:
    """Принятый шаблон для обращения или None."""
    if situation not in REPLACEABLE_SITUATIONS or not accepted.has_any(kind, lang):
        return None
    match = nearest_cluster(kind, lang, player_text, rule_refs_raw)
    if match is None or match.similarity < MIN_CENTROID_SIM:
        return None
    tpl = accepted.get(match.cluster)
    return (tpl, match) if tpl else None
