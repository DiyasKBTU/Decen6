"""Защитные проверки черновика и решение «нужен ли человек».

Работают одинаково для шаблонов (этап 1) и для текста LLM (этап 4):
- каждая ссылка в тексте существует и стоит в cited;
- запрещённые обещания (разбан, возврат, наказание другого игрока) -> needs_human.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config
from .rules import RuleIndex
from .taxonomy import Taxonomy

# Нельзя никогда: разбан, снятие наказания, возврат, компенсация
HARD_FORBIDDEN = re.compile(
    "|".join([
        r"разбан", r"сним(ем|ут)\s+(бан|наказ|мут|варн|блокир)", r"наказание\s+(будет\s+)?снят",
        r"верн(ём|ем|ут)\b", r"возмест", r"компенсир", r"восстанов(им|ят)\s+(имущ|машин|деньг|аккаунт)",
        r"\bunban", r"refund", r"will\s+be\s+(restored|returned|refunded|lifted)", r"we\s+will\s+(return|restore|refund|lift)",
        r"qaytar(amiz|ib\s+beramiz)", r"jazo\s+olib\s+tashlan", r"қайтарамыз", r"қалпына\s+келтіреміз",
        r"devolver(emos|á)", r"reembols",
    ]),
    re.I,
)
# Можно только с needs_human: наказание другого игрока
PUNISHMENT_PROMISE = re.compile(
    "|".join([
        r"получит\s+(бан|мут|наказ|блокир|деморган|предупр|джайл)", r"будет\s+(наказан|заблокирован|забанен)",
        r"will\s+be\s+(banned|punished|muted)", r"will\s+receive", r"jazo\s+oladi", r"жаза\s+қолданылады",
        r"recibirá", r"receberá",
    ]),
    re.I,
)
FORBIDDEN_PROMISES = re.compile(f"{HARD_FORBIDDEN.pattern}|{PUNISHMENT_PROMISE.pattern}", re.I)


@dataclass
class PolicyResult:
    needs_human: bool
    reasons: list[str] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)  # что пришлось исправить


def check_citations(text: str, cited: list[str], rules: RuleIndex) -> tuple[list[str], list[str]]:
    """Оставляет только существующие пункты и добавляет упомянутые в тексте.

    Возвращает (итоговый cited, список проблем). Несуществующий пункт в тексте —
    проблема, из-за которой черновик нельзя отправлять автоматически.
    """
    problems: list[str] = []
    final = [c for c in cited if rules.exists(c)]
    problems += [f"несуществующий пункт в cited: {c}" for c in cited if not rules.exists(c)]
    for pid in rules.parse_refs(text):
        if pid not in final:
            final.append(pid)
    for m in re.finditer(r"(?:п\.|пункт\w*|rule|regla|regra)\s*(\d{1,2}(?:\.\d{1,2}){1,2})", text, re.I):
        num = m.group(1)
        # окно может захватить следующую ссылку («п. 9.9 ... и п. 1.1 GZ») — нужен пункт именно с этим номером
        window = text[m.start(): m.end() + 30]
        if not any(pid.split()[-1] == num for pid in rules.parse_refs(window)):
            problems.append(f"ссылка на несуществующий пункт в тексте: {num}")
    return final, problems


def decide(
    *,
    category: str,
    situation: str,
    confidence: float,
    text: str,
    cited: list[str],
    promises_punishment: bool,
    threat: bool,
    self_harm: bool,
    rules: RuleIndex,
    tax: Taxonomy,
) -> PolicyResult:
    reasons: list[str] = []
    final_cited, problems = check_citations(text, cited, rules)

    if category in tax.needs_human_always:
        reasons.append({
            "report.payment": "платежи всегда разбирает человек",
            "report.account": "доступ к аккаунту и пропажа имущества всегда идут человеку",
            "complaint.admin": "апелляции и жалобы на администраторов всегда идут человеку",
        }.get(category, "категория всегда требует человека"))
    if promises_punishment or situation == "complaint_violation":
        reasons.append("решение о наказании подтверждает администратор")
    elif FORBIDDEN_PROMISES.search(text):
        reasons.append("в черновике есть обещание, которое может дать только администратор")
    if threat or self_harm:
        reasons.append("угроза или риск для игрока — срочно человеку")
    if situation == "complaint_duplicate":
        reasons.append("похоже на дубль жалобы — склейку подтверждает администратор")
    if situation == "complaint_need_video":
        reasons.append("достаточно ли скриншотов, решает администратор (примерно в 10% случаев их хватает)")
    if situation in {"report_help", "report_complaint", "report_question", "complaint_review", "report_event_teleport"}:
        reasons.append("нужно действие администратора в игре или проверка доказательств")
    if problems:
        reasons.append("проблема со ссылками на правила: " + "; ".join(problems))
    if confidence < config.MIN_AUTO_CONFIDENCE and not reasons:
        reasons.append(f"низкая уверенность классификации ({confidence:.2f})")

    return PolicyResult(needs_human=bool(reasons), reasons=reasons, cited=final_cited, violations=problems)
