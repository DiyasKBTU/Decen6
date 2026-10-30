"""Этап 3: ML-классификатор категории поверх правил.

Признаки: TF-IDF по символам текста игрока и поля «пункт правил» + решения правил
этапа 1 (категория, ситуация, сигналы). Обучен на слабой разметке из ответов админов
(scripts/train_classifier.py), лежит в artifacts/. Нет файла модели — работают правила.
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from . import config
from .classify import Classification
from .extract import Facts

log = logging.getLogger(__name__)

_NICK = re.compile(r"\b[A-Z][a-z]{1,15}_[A-Z][a-z]{1,15}\d{0,3}\b")


def clean_text(text: str) -> str:
    text = _NICK.sub(" ник ", text)
    text = re.sub(r"<url>", " url ", text)
    text = re.sub(r"\d", "0", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def refs_text(raw: str) -> str:
    """Поле «пункт правил»; префикс — чтобы у репортов (поле пустое) словарь не был пустым."""
    return "refs " + clean_text(raw or "")


def rule_features(kind: str, cls: Classification, facts: Facts, lang: str, accused_replied: bool, text: str) -> dict:
    n = len(text)
    return {
        f"rule_cat={cls.category}": 1,
        f"rule_sit={cls.situation}": 1,
        f"rule_ref={cls.primary_ref or cls.rp_term or '-'}": 1,
        f"evidence={facts.evidence_kind}": 1,
        f"images={min(facts.images, 3)}": 1,
        f"lang={lang}": 1,
        "accused_replied": int(accused_replied),
        "has_refs": int(bool(facts.rule_refs)),
        f"len={'s' if n < 15 else 'm' if n < 60 else 'l'}": 1,
        "rule_conf": cls.confidence,
        **{f"claimed={r}": 1 for r in facts.rule_refs[:3]},
    }


@dataclass
class Prediction:
    category: str
    prob: float
    probs: dict[str, float]


class KindModel:
    """Модель для одного типа обращения (категории жалоб и репортов не пересекаются)."""

    def __init__(self, char_vec, refs_vec, dict_vec, clf):
        self.char_vec, self.refs_vec, self.dict_vec, self.clf = char_vec, refs_vec, dict_vec, clf

    def matrix(self, texts: list[str], refs: list[str], feats: list[dict]):
        return sp.hstack([
            self.char_vec.transform([clean_text(t) for t in texts]),
            self.refs_vec.transform([refs_text(r) for r in refs]),
            self.dict_vec.transform(feats),
        ]).tocsr()

    def predict(self, text: str, refs_raw: str, feats: dict) -> Prediction:
        proba = self.clf.predict_proba(self.matrix([text], [refs_raw], [feats]))[0]
        i = int(np.argmax(proba))
        return Prediction(self.clf.classes_[i], float(proba[i]), dict(zip(self.clf.classes_, map(float, proba))))


@lru_cache(maxsize=1)
def get_models() -> dict[str, KindModel] | None:
    if config.ML_MODE in ("0", "off", "false"):
        return None
    path = config.CLASSIFIER_PATH
    if not path.exists():
        log.warning("модель классификатора не найдена (%s): работают правила; make train", path)
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return {kind: KindModel(**parts) for kind, parts in payload["models"].items()}
