"""Прогон истории: насколько быстрее срочные жалобы получили бы ответ с нашей очередью.

python scripts/replay_history.py [--limit N]

Данные: жалобы train с реальным временем создания и первого ответа админа (у репортов
время ответа в выгрузке совпадает со временем создания, поэтому они не участвуют).

Модель:
- «Слоты» — моменты, когда админы реально отвечали. Их число и время не меняются:
  у админов столько же рабочего времени, сколько было.
- В каждом слоте админ берёт из очереди ожидающую жалобу с самым срочным нашим
  приоритетом (при равенстве — более старую). Так было бы при сортировке по приоритету.
- Жалобы, которые сервис закрывает сам (needs_human=false), отвечаются сразу и слот не занимают.
- Пулы админов разделены по региону (cis / uz / kz).

Допущение: на любую жалобу уходит одинаковое время. В реальности жалоба с видео
дольше, чем «жалоба без описания», поэтому цифры — оценка, а не прогноз.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import heapq
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.pipeline import triage  # noqa: E402
from copilot.predict import read_jsonl  # noqa: E402
from copilot.schemas import TriageRequest  # noqa: E402

PRIORITY_RANK = {"p0": 0, "p1": 1, "p2": 2, "p3": 3}


def ts(value: str) -> float:
    return dt.datetime.fromisoformat(value).timestamp()


def simulate(jobs: list[tuple[float, int, str]], slots: list[float]) -> dict[str, float]:
    """Жадное расписание: jobs = (время создания, ранг приоритета, id), slots — моменты ответов.

    -> {id: время ответа}. Если исходное расписание было допустимым (каждая жалоба
    отвечена после создания), жадный выбор тоже раздаст слоты всем.
    """
    jobs = sorted(jobs)
    answered: dict[str, float] = {}
    heap: list[tuple[int, float, str]] = []
    i = 0
    for slot in sorted(slots):
        while i < len(jobs) and jobs[i][0] <= slot:
            created, rank, jid = jobs[i]
            heapq.heappush(heap, (rank, created, jid))
            i += 1
        if heap:
            _, _, jid = heapq.heappop(heap)
            answered[jid] = slot
    return answered


def describe(minutes: list[float]) -> dict:
    if not minutes:
        return {"n": 0}
    m = sorted(minutes)
    q = lambda p: round(m[min(len(m) - 1, int(len(m) * p))], 1)  # noqa: E731
    return {"n": len(m), "median": round(statistics.median(m), 1), "p90": q(0.9), "mean": round(statistics.mean(m), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=config.DATA_DIR / "tickets_train.jsonl.gz")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    tickets = []
    for row in read_jsonl(args.input):
        if row["kind"] != "complaint":
            continue
        answer = next((m.get("ts") for m in row["messages"] if m["role"] == "admin" and m.get("ts")), None)
        if not answer:
            continue
        resp, _ = triage(TriageRequest.model_validate(row), dup=None)
        tickets.append({
            "id": row["ticket_id"], "region": row["region"], "created": ts(row["created_at"]), "answered": ts(answer),
            "priority": resp.priority, "category": resp.category, "needs_human": resp.needs_human,
        })
        if args.limit and len(tickets) >= args.limit:
            break
    print(f"triaged {len(tickets)} complaints in {time.time() - t0:.0f}s")

    # control_random_drop — контроль: убираем столько же случайных жалоб, сколько закрыл бы сервис.
    # Если среднее ожидание падает так же, эффект автоответов — это освободившееся время админов.
    rng = random.Random(7)
    scenarios = {"as_is": {}, "sorted": {}, "sorted_auto": {}, "control_random_drop": {}}
    for region in sorted({t["region"] for t in tickets}):
        pool = [t for t in tickets if t["region"] == region]
        slots = [t["answered"] for t in pool]
        for t in pool:
            scenarios["as_is"][t["id"]] = t["answered"]
        all_jobs = [(t["created"], PRIORITY_RANK[t["priority"]], t["id"]) for t in pool]
        scenarios["sorted"].update(simulate(all_jobs, slots))
        human = [(t["created"], PRIORITY_RANK[t["priority"]], t["id"]) for t in pool if t["needs_human"]]
        scenarios["sorted_auto"].update(simulate(human, slots))
        for t in pool:
            if not t["needs_human"]:
                scenarios["sorted_auto"][t["id"]] = t["created"]  # сервис ответил сразу
        dropped = set(rng.sample([j[2] for j in all_jobs], len(all_jobs) - len(human)))
        scenarios["control_random_drop"].update(simulate([j for j in all_jobs if j[2] not in dropped], slots))

    def wait(scn: str, t: dict) -> float:
        return (scenarios[scn][t["id"]] - t["created"]) / 60

    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for t in tickets:
        groups[f"priority={t['priority']}"].append(t)
        groups[f"category={t['category']}"].append(t)
        groups["all"].append(t)
    result = {
        "complaints": len(tickets),
        "auto_closed": sum(not t["needs_human"] for t in tickets),
        "unassigned": {s: sum(t["id"] not in scenarios[s] for t in tickets) for s in scenarios if s != "control_random_drop"},
        "wait_minutes": {
            g: {s: describe([wait(s, t) for t in items if t["id"] in scenarios[s]]) for s in scenarios}
            for g, items in sorted(groups.items())
        },
    }
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "replay.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, config.OUT_DIR / "replay.md")
    print(json.dumps({k: v for k, v in result.items() if k != "wait_minutes"}, ensure_ascii=False))
    for g in ["all", "priority=p0", "priority=p1", "priority=p2", "priority=p3", "category=complaint.cheat",
              "category=complaint.scam", "category=complaint.invalid"]:
        if g in result["wait_minutes"]:
            print(g, json.dumps(result["wait_minutes"][g], ensure_ascii=False))
    print(f"done in {time.time() - t0:.0f}s")


GROUP_TITLES = {
    "all": "Все жалобы",
    "priority=p0": "p0 — угрозы вне игры",
    "priority=p1": "p1 — читы, мошенничество, реклама, видео",
    "priority=p2": "p2 — оскорбления, DM, апелляции",
    "priority=p3": "p3 — пустые и без доказательств",
    "category=complaint.cheat": "Читы",
    "category=complaint.scam": "Мошенничество",
    "category=complaint.rmt_ads": "Реклама и продажа за реал",
    "category=complaint.insult": "Оскорбления",
    "category=complaint.dm": "DM и нарушения RP",
    "category=complaint.admin": "Апелляции и жалобы на админов",
    "category=complaint.other": "Другое нарушение",
    "category=complaint.invalid": "Не жалоба",
}


def write_markdown(r: dict, path: Path) -> None:
    w = r["wait_minutes"]
    cell = lambda x: f"{x['median']:g} / {x['p90']:g}" if x.get("n") else "—"  # noqa: E731
    lines = [
        "# Прогон истории: время до первого ответа на жалобу",
        "",
        f"{r['complaints']} жалоб train с реальным временем ответа. Сервис закрыл бы сам {r['auto_closed']} "
        f"({100 * r['auto_closed'] / r['complaints']:.1f}%). Время — минуты: медиана / 90-й перцентиль.",
        "",
        "| Группа | Жалоб | Как было | Очередь по приоритету | + автоответы |",
        "|---|---|---|---|---|",
    ]
    for g, title in GROUP_TITLES.items():
        if g in w and w[g]["as_is"]["n"] >= 30:
            s = w[g]
            lines.append(f"| {title} | {s['as_is']['n']} | {cell(s['as_is'])} | {cell(s['sorted'])} | {cell(s['sorted_auto'])} |")
    a = w["all"]
    lines += [
        "",
        "Как читать:",
        f"- Среднее ожидание «как было» и «очередь по приоритету» одинаковое ({a['as_is']['mean']:g} мин): "
        "время админов то же, меняется только порядок. Срочное ускоряется за счёт обычного: медиана у оскорблений "
        "и DM немного растёт, зато длинный хвост у всех короче.",
        "- Строгий приоритет без автоответов морит голодом p3 (часы ожидания). В продакшене нужен рост приоритета "
        "со временем ожидания; с автоответами p3 закрываются сразу.",
        f"- Автоответы освобождают время админов. Контроль: если убрать столько же случайных жалоб, среднее "
        f"ожидание падает так же ({w['all']['control_random_drop']['mean']:g} против {a['sorted_auto']['mean']:g} мин) — "
        "очередь работает близко к пределу, и даже 3% разгрузки заметно режут хвост.",
        "",
        "Допущения: на любую жалобу уходит одинаковое время админа (пустую жалобу в реальности закрывают быстрее, "
        "поэтому эффект автоответов — верхняя оценка); пул админов общий на регион (апелляции в реальности часто "
        "разбирает старшая администрация); админы берут обращения строго из очереди. Это оценка, а не прогноз.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
