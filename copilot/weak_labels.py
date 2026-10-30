"""Слабая разметка категорий из ответов админов — только для обучения и оценки.

Сервис этим модулем не пользуется: ответа админа на входе нет. Метка ставится,
только когда ответ однозначно выдаёт категорию:
- апелляция («наказание выдано верно», вердикт appeal_*) -> complaint.admin;
- наказание с пунктом нарушения -> категория этого пункта по taxonomy.json;
- «неверно оформлена» -> complaint.invalid, только если причина — нет описания,
  вопрос вместо жалобы или общие слова (ЖБ 1.1.2, 1.2.3, 1.3.3, 1.3.4, 1.1.3).
  Остальные причины (48 часов, грубый текст, не тот заявитель) говорят о форме,
  а не о содержании, и метку не дают.
"""

from __future__ import annotations

import re

from .rules import RuleIndex
from .taxonomy import Taxonomy

TERM_DM = re.compile(r"\b(dm|дм|sk|ск|db|дб|tk|тк|nrp|нрп|nonrp|деморган|джайл|jail|demorgan)\b", re.I)
APPEAL_ANSWER = re.compile(r"наказание (было )?(выдано|выдан) верно|jazo to‘g‘ri|наказание будет снято|наказание снято", re.I)
ADMIN_TALK = re.compile(r"с администратором .{0,40}(беседа|беседу|дисциплинар)|administrator .{0,40}suhbat", re.I)
INVALID_REASONS = {"ЖБ 1.1.2", "ЖБ 1.2.3", "ЖБ 1.3.3", "ЖБ 1.3.4", "ЖБ 1.1.3"}


def admin_text(row: dict) -> str:
    return "\n".join(m["text"] for m in row["messages"] if m["role"] == "admin")


def label(row: dict, rules: RuleIndex, tax: Taxonomy) -> str | None:
    verdict = row.get("verdict")
    answer = admin_text(row)
    if row["kind"] == "complaint":
        if verdict in ("appeal_approved", "appeal_rejected") or APPEAL_ANSWER.search(answer) or ADMIN_TALK.search(answer):
            return "complaint.admin"
        if verdict == "invalid_format":
            refs = {r for r in rules.parse_refs(answer) if r.startswith("ЖБ")}
            return "complaint.invalid" if refs & INVALID_REASONS else None
        if verdict not in ("approved", "other", "already_punished"):
            return None
        text = f"{row.get('violation') or ''}\n{answer}"
        for ref in rules.parse_refs(text):
            if ref.startswith("ЖБ"):
                continue
            return tax.category_for_ref(ref, "complaint") or "complaint.other"
        if TERM_DM.search(text):
            return "complaint.dm"
        return None

    a = answer.lower()
    if "эвакуатор" in a:
        return "report.help"
    if re.search(r"при последующем оффтопе|оффтоп", a):
        return "report.offtopic"
    if re.search(r"уточните (ваш )?(вопрос|жалобу)", a):
        return "report.offtopic"
    if re.search(r"при наличии доказательств|слежу за игроком|не вмешиваемся в рп|не вмешивается в рп", a):
        return "report.complaint"
    if re.search(r"телепорт|не предоставляем|не оказываем|не телепортируем", a):
        return "report.help"
    return None
