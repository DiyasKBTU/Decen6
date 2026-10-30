"""Индекс правил: канонические идентификаторы пунктов, белый список, разбор ссылок.

Номера пунктов не уникальны («1.1» есть в 10 разделах), поэтому у каждого раздела
свой префикс. Общие правила идут без префикса, как в примере контракта («2.1»);
«ЖБ» и «GZ» взяты из taxonomy.json, остальные — из того, как пишут админы.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import config

# id раздела в rules.json -> префикс канонической ссылки
SECTION_PREFIX = {
    30: "",  # Общие правила игры (ОП)
    31: "ЖБ",  # Правила жалоб (админы пишут ППЖ)
    32: "GZ",  # Зелёная зона
    33: "ПЛЗ",  # Лидеры и заместители
    34: "ПГО",  # Государственные организации
    35: "ПГВД",  # Гос. волна и департамент
    36: "FW",  # Семейные мероприятия
    40: "ОПГ",
    41: "BW",  # Война за бизнес
    42: "FURA",  # Битва за фургон
}
RP_TERMS_SECTION = 43

# Как разделы называют игроки и админы -> наш префикс
CODE_ALIASES = {
    "ОП": "", "OP": "", "ОР": "", "OП": "", "ОПР": "",
    "ЖБ": "ЖБ", "ППЖ": "ЖБ", "ПЖ": "ЖБ",
    "GZ": "GZ", "ГЗ": "GZ", "ЗЗ": "GZ", "ZZ": "GZ",
    "ПЛЗ": "ПЛЗ", "ОПЛЗ": "ПЛЗ",
    "ПГО": "ПГО", "PGO": "ПГО", "ГОС": "ПГО", "ГОСС": "ПГО", "НПГО": "ПГО",
    "ПГВД": "ПГВД",
    "FW": "FW", "ФВ": "FW", "FV": "FW",
    "ОПГ": "ОПГ", "OPG": "ОПГ", "ПОПГ": "ОПГ",
    "BW": "BW",
    "FURA": "FURA", "ФУРА": "FURA",
}

# Названия разделов для текста черновика
SECTION_TITLES = {
    "": {"ru": "общих правил", "kk": "жалпы ережелердің", "uz": "Umumiy qoidalar (OP)",
         "en": "of the General Rules", "es": "de las Reglas Generales", "pt": "das Regras Gerais"},
    "ЖБ": {"ru": "правил подачи жалоб", "kk": "шағым беру ережелерінің", "uz": "Shikoyat berish qoidalari",
           "en": "of the Complaint Rules", "es": "de las Reglas de Quejas", "pt": "das Regras de Denúncias"},
    "GZ": {"ru": "правил Зелёной зоны", "kk": "Жасыл аймақ ережелерінің", "uz": "Yashil zona (GZ) qoidalari",
           "en": "of the Green Zone Rules", "es": "de las Reglas de la Zona Verde", "pt": "das Regras da Zona Verde"},
}

_REF_NUM = r"\d{1,2}(?:\.\d{1,2}){1,2}"
_LETTER = r"A-Za-zА-Яа-яЁё"
_CODE = rf"[{_LETTER}]{{2,4}}(?![{_LETTER}])"
# «2.1 ОП», «2.1оп», «1.1 gz», «п. 1.2.7 правил подачи жалоб», «GZ 1.1»
_RE_NUM_CODE = re.compile(rf"(?<![\d.])({_REF_NUM})(?![\d])\s*[-–]?\s*({_CODE})?", re.U)
_RE_CODE_NUM = re.compile(rf"(?<![{_LETTER}])({_CODE})\s*({_REF_NUM})(?![\d])", re.U)
_RE_JB_WORDS = re.compile(r"правил[а-я]*\s+(подачи\s+)?жалоб|ppj|ppzh|shikoyat", re.I)
_RE_SANCTION = re.compile(
    r"\|\s*((?:[^|:]{0,40}:\s*)?(?:MUTE|WARN|DEMORGAN|JAIL|BAN|Блокировк|Устн|Письменн|Предупрежд|"
    r"Дисквал|Расформ|Снят|SPAWN|Откат|Наказание согласно)[^|]{0,110})",
    re.I,
)


@dataclass(frozen=True)
class RulePoint:
    id: str  # каноническая ссылка: «2.1», «ЖБ 1.2.7», «GZ 1.1»
    prefix: str
    ref: str
    section_id: int
    section_name: str
    text: str

    @property
    def short(self) -> str:
        """Суть пункта без блока наказаний, до ~160 символов."""
        body = re.sub(r"\s*\[[^\]]*\]", "", self.text)
        body = body.split(" | ")[0].strip()
        body = re.split(r"(?<=[.!?])\s", body, maxsplit=1)[0]
        return body if len(body) <= 160 else body[:157].rsplit(" ", 1)[0] + "…"

    @property
    def punishment(self) -> str | None:
        """Санкция из текста пункта («Нарушение: MUTE ...»), если указана."""
        m = _RE_SANCTION.search(self.text)
        if not m:
            return None
        # в конце пункта иногда приклеен заголовок следующего раздела: «III. ПРАВИЛА ...»
        return re.split(r"\s+[IVX]{1,4}\.\s", m.group(1))[0].strip().rstrip(".")


class RuleIndex:
    def __init__(self, path: Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.points: dict[str, RulePoint] = {}
        self.rp_terms: dict[str, str] = {}
        for section in data["rules"]:
            sid = section["id"]
            if sid == RP_TERMS_SECTION:
                self._load_terms(section["text"])
                continue
            prefix = SECTION_PREFIX.get(sid)
            if prefix is None:
                continue
            for p in section.get("points", []):
                pid = f"{prefix} {p['ref']}".strip()
                self.points[pid] = RulePoint(pid, prefix, p["ref"], sid, section["category"], p["text"])

    def _load_terms(self, text: str) -> None:
        for m in re.finditer(r"•\s*[^\n(]*\(сокращенно\s+([^)]+)\)\s*[-—–]\s*([^\n]+)", text):
            self.rp_terms[m.group(1).strip().upper()] = m.group(2).strip()

    def exists(self, pid: str) -> bool:
        return pid in self.points

    def get(self, pid: str) -> RulePoint | None:
        return self.points.get(pid)

    def canonical(self, num: str, code: str | None = None, context: str = "") -> str | None:
        """Номер + код раздела -> существующая каноническая ссылка или None."""
        if code:
            prefix = CODE_ALIASES.get(code.upper())
            if prefix is None:
                prefix = ""  # «2.1 оск» — код не раздел, считаем общими правилами
        elif _RE_JB_WORDS.search(context):
            prefix = "ЖБ"
        else:
            prefix = ""
        pid = f"{prefix} {num}".strip()
        return pid if pid in self.points else None

    def parse_refs(self, text: str) -> list[str]:
        """Все существующие пункты, на которые ссылается текст, в порядке появления."""
        if not text:
            return []
        found: list[tuple[int, str]] = []
        taken: list[tuple[int, int]] = []
        for m in _RE_CODE_NUM.finditer(text):
            if m.group(1).upper() in CODE_ALIASES:
                pid = self.canonical(m.group(2), m.group(1))
                if pid:
                    found.append((m.start(), pid))
                    taken.append(m.span(2))
        for m in _RE_NUM_CODE.finditer(text):
            if any(a <= m.start(1) < b for a, b in taken):
                continue
            tail = text[m.end(1): m.end(1) + 40]
            pid = self.canonical(m.group(1), m.group(2), context=tail)
            if pid:
                found.append((m.start(), pid))
        result: list[str] = []
        for _, pid in sorted(found):
            if pid not in result:
                result.append(pid)
        return result


@lru_cache(maxsize=1)
def get_index() -> RuleIndex:
    return RuleIndex(config.RULES_PATH)
