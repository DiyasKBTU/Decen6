"""Обязательный шаг RAG: перед любым черновиком сервис находит, на что опереться.

Для каждого обращения собирается «основание ответа» (Evidence):
- пункты правил — гибридный поиск с подсказкой категории (retrieval.RuleRetriever);
- похожие прошлые обращения с ответами админов — база знаний типовых решений;
- термины RP из раздела «RP термины» (DM, SK, ...), если они упомянуты.

Черновик (шаблон, принятый шаблон или LLM) ссылается только на найденные пункты, подходящие
к категории. Основание сохраняется в meta["evidence"]: админ видит его в карточке, LLM получает
его в запросе. Готовые ответы из базы знаний даёт библиотека шаблонов, которые админ проверил
и принял (template_library): сырые прошлые ответы для автоответа слишком шумные (проверено на valid).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config, templates
from .classify import Classification
from .retrieval import Example, RuleHit, get_history_retriever, get_rule_retriever
from .rules import RuleIndex
from .taxonomy import Taxonomy

GZ_WORDS = re.compile(r"\bзз\b|\bzz\b|\bgz\b|\bгз\b|зел[её]н\w* зон|yashil zona", re.I)
TERM_PATTERNS = {
    "DM": r"\bдм\b|\bdm\b", "DB": r"\bдб\b|\bdb\b", "SK": r"\bск\b|\bsk\b", "TK": r"\bтк\b|\btk\b",
    "RK": r"\bрк\b|\brk\b", "MG": r"\bмг\b|\bmg\b", "PG": r"\bпг\b|\bpg\b", "NONRP": r"\bнрп\b|\bnrp\b|non ?rp",
}


@dataclass
class Evidence:
    rules: list[RuleHit] = field(default_factory=list)
    prior: set[str] = field(default_factory=set)
    examples: list[Example] = field(default_factory=list)
    terms: dict[str, str] = field(default_factory=dict)

    def to_meta(self, cited: list[str], rules: RuleIndex) -> dict:
        return {
            "rules": [{"id": h.id, "score": h.score, "short": rules.get(h.id).short, "cited": h.id in cited,
                       "fits_category": h.id in self.prior} for h in self.rules],
            "examples": [{"player": e.player[:200], "admin": e.admin[:300], "score": round(e.score, 2), "lang": e.lang,
                          "verdict": e.verdict} for e in self.examples[:3]],
            "terms": self.terms,
        }


def category_prior(cls: Classification, text: str, tax: Taxonomy, rules: RuleIndex) -> set[str]:
    """Пункты, подходящие к категории и ситуации (их поиск поднимает выше)."""
    prior: set[str] = set()
    for ref in tax.rules_of(cls.category):
        if ref == "GZ":
            if GZ_WORDS.search(text):
                prior |= {pid for pid in rules.points if pid.startswith("GZ 1.")}
        elif rules.exists(ref):
            prior.add(ref)
    for ref in (cls.primary_ref, templates.DEFAULT_REFS.get(cls.situation), templates.SECOND_REFS.get(cls.situation)):
        if ref and rules.exists(ref):
            prior.add(ref)
    return prior


def gather(kind: str, text: str, rule_refs_raw: str | None, cls: Classification, reply_lang: str,
           rules: RuleIndex, tax: Taxonomy) -> Evidence:
    prior = category_prior(cls, text, tax, rules)
    query = f"{text} {rule_refs_raw or ''}"
    retriever = get_rule_retriever()
    hits = retriever.search(query, k=5, prior=prior)
    # пункты, которые требует ситуация (например, правила подачи жалоб для «нужно видео»),
    # всегда есть в основании — со своей оценкой из полного ранжирования
    note = "1.4" if cls.category in ("complaint.scam", "report.account") else None  # «имущество не восстанавливается»
    required = [r for r in (cls.primary_ref, templates.DEFAULT_REFS.get(cls.situation),
                            templates.SECOND_REFS.get(cls.situation), note) if r and rules.exists(r)]
    missing = [r for r in required if r not in {h.id for h in hits}]
    if missing:
        full = {h.id: h for h in retriever.search(query, k=len(retriever.ids), prior=prior, min_score=-1)}
        hits += [full[r] for r in missing if r in full]
    history = get_history_retriever() if config.RAG_EXAMPLES else None
    examples = history.search(text, kind, reply_lang, k=3) if history else []
    terms = {t: rules.rp_terms.get(t, "").split(" | ")[0].strip() for t, rx in TERM_PATTERNS.items()
             if re.search(rx, text, re.I) and t in rules.rp_terms}
    return Evidence(rules=hits, prior=prior, examples=examples, terms=terms)


def choose_citation(cls: Classification, ev: Evidence) -> str | None:
    """Пункт для черновика с наказанием: лучший найденный среди подходящих к категории.

    На valid выбор поиском и прежней таблицей «категория -> пункт» совпадает с пунктом админа
    одинаково часто (81,6% и 82%), поэтому ссылку даёт поиск — так черновик опирается на найденное.
    """
    if not cls.primary_ref:
        return cls.primary_ref
    fitting = [h for h in ev.rules if h.id in ev.prior and not h.id.startswith("ЖБ")]
    if not fitting:
        return cls.primary_ref
    current = next((h.score for h in fitting if h.id == cls.primary_ref), None)
    best = fitting[0]
    if best.id != cls.primary_ref and (current is None or best.score >= current + 0.05):
        return best.id
    return cls.primary_ref
