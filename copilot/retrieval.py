"""Поиск по базе знаний (RAG): пункты правил и прошлые ответы админов.

1. Пункты правил — гибридный поиск:
   - сходство по фрагментам слов (устойчиво к опечаткам) и по целым словам;
   - у пунктов есть ключевые слова игроков («софт, чит, аим» у 5.2), а сленг в запросе
     расшифровывается («дм» -> «DeathMatch убийство без RP-причины»);
   - общие правила, правила жалоб и Зелёной зоны важнее разделов для лидеров, госорганизаций
     и мероприятий — если игрок сам не упомянул такой раздел;
   - пункт, названный игроком, и пункты, подходящие к категории, поднимаются выше.
2. Прошлые обращения с ответами админов — база знаний типовых ответов, в том числе на
   казахском и узбекском. Индекс строится из train (scripts/build_history_index.py) и лежит
   в artifacts/ — в репозиторий не попадает, как и сами данные.

Качество поиска: scripts/eval_retrieval.py.
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from . import config
from .rules import RuleIndex, get_index

log = logging.getLogger(__name__)

SLANG = {
    r"\bдм\b|\bdm\b": "DeathMatch убийство без основательной RP причины",
    r"\bдб\b|\bdb\b": "DriveBy убийство за рулём транспорта таран",
    r"\bск\b|\bsk\b": "Spawn Kill убийство на месте появления",
    r"\bтк\b|\btk\b": "Team Kill убийство членов своей организации",
    r"\bрк\b|\brk\b": "Revenge Kill убийство из мести",
    r"\bзз\b|\bzz\b|\bгз\b|\bgz\b": "Зелёная Зона убийства урон",
    r"\bнрп\b|\bnrp\b|non ?rp|\bурп\b": "NonRP нарушение RP режима",
    r"\bмг\b|\bmg\b": "MetaGaming OOC информация в IC",
    r"\bпг\b|\bpg\b": "PowerGaming преувеличение возможностей",
    r"\bоск|\bosk\b|so[‘']?k|haqorat": "оскорбление личности игрока родственников",
    r"\bсофт|\bчит|soft|cheat|\bаим": "стороннее программное обеспечение читы преимущество",
    r"скам|scam|обман|\bкинул|aldab": "мошенничество махинации обман игроков",
    r"реклам|\bтг\b|телеграм|за реал": "реклама сделки за реальные деньги",
    r"опру|не соглас|наказан|снять|сними": "обжалование наказания жалоба на администратора",
    r"\bтп\b|телепорт": "репорт не по назначению телепорт",
    r"наручник|мешок|в[её]р[её]вк|похит": "криминальные действия похищение верёвки мешки",
}

# Как игроки называют нарушения: добавляется к тексту пункта при индексации
RULE_KEYWORDS = {
    "2.1": "оскорбление оск оскорбил мат матерится родня родные родных мать мама семья упоминание родни "
           "унижение haqorat sokindi onadan osk",
    "2.2": "пропаганда наркотики запрещённые темы экстремизм",
    "2.3": "реклама телеграм тг дискорд ссылка другой сервер проект reklama",
    "2.4": "расизм нация национальность религия ислам ксенофобия политика",
    "2.5": "музыка в микрофон крики шум войс",
    "2.6": "угрозы деанон доксинг личная информация ooc найду в реале",
    "3.1": "репорт оффтоп флуд телепорт тп просьба валюта",
    "3.2": "хамство администрации оскорбил админа клевета обман админа",
    "3.4": "оспаривание действий администрации жалоба на админа",
    "4.1": "мошенничество скам обман кинул развел не отдал деньги машину махинации aldadi",
    "4.2": "вымогательство шантаж требует деньги",
    "4.4": "продажа за реальные деньги реал аккаунт продаёт сделка",
    "5.2": "софт чит читы читер аим вх wallhack флай летает сборка багоюз стороннее по cheat soft",
    "5.3": "уход от рп вышел из игры ливнул",
    "5.4": "дом квартира у дома убил у входа территория дома",
    "5.5": "стрим стример стримснайп",
    "5.6": "аморальные действия унижение",
    "1.6": "ник никнейм оскорбительный ник",
    "GZ 1.1": "зз зелёная зона гз убил в зз урон дм в зз стрельба больница банк",
    "GZ 1.2": "зз похищение ограбление верёвка мешок наручники связал в зз",
    "GZ 1.3": "уход от рп в зз",
    "GZ 1.4": "вывел из зз верёвкой",
    "GZ 1.5": "зашёл в зз во время перестрелки",
}
CORE_PREFIXES = {"", "ЖБ", "GZ"}
OTHER_SECTION_WEIGHT = 0.55
# Если игрок упомянул раздел, он важен наравне с общими правилами
SECTION_MENTIONS = {
    "FW": r"\bфв\b|\bfw\b|\bfv\b|семейн",
    "ОПГ": r"\bопг\b|\bopg\b",
    "ПГО": r"\bпго\b|\bpgo\b|\bгос\b|госс",
    "ПЛЗ": r"\bплз\b|\bоплз\b|лидер",
    "BW": r"\bbw\b|бизвар|войн\w* за бизнес",
    "FURA": r"\bфура\b|\bfura\b|фургон",
    "ПГВД": r"\bпгвд\b|гос\.? ?волн",
}
CLAIMED_BOOST = 0.3
PRIOR_BOOST = 0.25


@dataclass
class RuleHit:
    id: str
    score: float


@dataclass
class Example:
    kind: str
    lang: str
    player: str
    admin: str
    verdict: str | None
    score: float


def expand_query(text: str) -> str:
    extra = [v for k, v in SLANG.items() if re.search(k, text, re.I)]
    return text + " " + " ".join(extra)


class CharRuleRetriever:
    """Первая версия: только сходство по фрагментам слов. Оставлена для сравнения в eval_retrieval."""

    def __init__(self, rules: RuleIndex):
        self.ids = list(rules.points)
        docs = [f"{p.section_name}. {p.text}" for p in rules.points.values()]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
        self.matrix = self.vec.fit_transform(docs)

    def search(self, text: str, k: int = 5, min_score: float = 0.08) -> list[RuleHit]:
        q = self.vec.transform([expand_query(text.lower())])
        scores = linear_kernel(q, self.matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [RuleHit(self.ids[i], float(scores[i])) for i in order if scores[i] >= min_score]


def is_heading(text: str) -> bool:
    """«ЖАЛОБА НЕ ПРИНИМАЕТСЯ К РАССМОТРЕНИЮ, ЕСЛИ:» — заголовок раздела, а не правило."""
    letters = [c for c in text if c.isalpha()]
    return len(text) < 80 and bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8


class RuleRetriever:
    def __init__(self, rules: RuleIndex):
        self.rules = rules
        points = {pid: p for pid, p in rules.points.items() if not is_heading(p.text)}
        self.ids = list(points)
        self.prefixes = [p.prefix for p in points.values()]
        docs = [f"{p.section_name}. {p.text} {RULE_KEYWORDS.get(pid, '')}".lower() for pid, p in points.items()]
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.word_vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w\w+\b", sublinear_tf=True)
        self.char_m = self.char_vec.fit_transform(docs)
        self.word_m = self.word_vec.fit_transform(docs)

    def search(self, text: str, k: int = 5, prior: set[str] | None = None, min_score: float = 0.05) -> list[RuleHit]:
        low = text.lower()
        q = expand_query(low)
        scores = 0.5 * linear_kernel(self.char_vec.transform([q]), self.char_m).ravel()
        scores += 0.5 * linear_kernel(self.word_vec.transform([q]), self.word_m).ravel()
        mentioned = {p for p, rx in SECTION_MENTIONS.items() if re.search(rx, low)}
        claimed = set(self.rules.parse_refs(text))
        for i, pid in enumerate(self.ids):
            if self.prefixes[i] not in CORE_PREFIXES and self.prefixes[i] not in mentioned:
                scores[i] *= OTHER_SECTION_WEIGHT
            if pid in claimed:
                scores[i] += CLAIMED_BOOST
            if prior and pid in prior:
                scores[i] += PRIOR_BOOST
        order = scores.argsort()[::-1][:k]
        return [RuleHit(self.ids[i], round(float(scores[i]), 3)) for i in order if scores[i] >= min_score]


class HistoryRetriever:
    """Похожие прошлые обращения того же типа; для kk/uz добавляет примеры на том же языке."""

    def __init__(self, payload: dict):
        self.vec = payload["vectorizer"]
        self.matrix = payload["matrix"]
        self.records = payload["records"]  # dict(kind, lang, player, admin, verdict)

    def search(self, text: str, kind: str, lang: str, k: int = 3) -> list[Example]:
        if not text.strip():
            return []
        scores = linear_kernel(self.vec.transform([text.lower()]), self.matrix).ravel()
        picked: list[int] = []

        seen_answers: set[str] = set()

        def take(pred, n):
            for i in scores.argsort()[::-1][:400]:
                if len([j for j in picked if pred(self.records[j])]) >= n or scores[i] <= 0:
                    break
                r = self.records[i]
                key = r["admin"][:120]
                if i not in picked and pred(r) and key not in seen_answers:
                    picked.append(i)
                    seen_answers.add(key)

        if lang in ("kk", "uz", "en", "es", "pt"):
            take(lambda r: r["kind"] == kind and r["lang"] == lang, 2)
        take(lambda r: r["kind"] == kind, k)
        return [Example(score=float(scores[i]), **self.records[i]) for i in picked[: k + 1]]


@lru_cache(maxsize=1)
def get_rule_retriever() -> RuleRetriever:
    return RuleRetriever(get_index())


@lru_cache(maxsize=1)
def get_history_retriever() -> HistoryRetriever | None:
    path = config.HISTORY_INDEX_PATH
    if not path.exists():
        log.warning("история ответов не найдена (%s): черновики без примеров; make index", path)
        return None
    with open(path, "rb") as f:
        return HistoryRetriever(pickle.load(f))
