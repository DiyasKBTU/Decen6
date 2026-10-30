"""Оценка на tickets_valid.jsonl: python scripts/eval_valid.py [--limit N]

Разметки категорий нет, поэтому категорию проверяем по «прокси-меткам» из ответа
админа (вердикт и пункт нарушения). Это не настоящая метка: она есть только у части
обращений и отражает решение админа, а не текст игрока. Честная оценка — на ручном
эталоне (этап 2).

Пишет out/eval_stage.json и out/report.md (с примерами — в репозиторий не коммитить).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from copilot import config, llm  # noqa: E402
from copilot.policy import FORBIDDEN_PROMISES  # noqa: E402
from copilot.predict import find_duplicates, percentile, process, read_jsonl  # noqa: E402
from copilot.rules import get_index  # noqa: E402
from copilot.taxonomy import get_taxonomy  # noqa: E402
from copilot.weak_labels import admin_text  # noqa: E402
from copilot.weak_labels import label as proxy_category  # noqa: E402

class EchoTemplateClient:
    """Фейковая модель для сухого прогона: возвращает текст шаблона из запроса."""

    def __init__(self):
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        content = kwargs["messages"][0]["content"]
        text = re.search(r"Черновик-шаблон \(можно взять за основу и улучшить\): (.*)", content).group(1)
        ids = re.findall(r'id "([^"]+)"', content)
        data = {"text": text, "cited": [i for i in ids if i.split()[-1] in text], "needs_human_hint": False, "hint_reason": ""}
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(data, ensure_ascii=False))],
                               usage=SimpleNamespace(input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0))


# Какое решение админа соответствует ситуации черновика
DECISION_VERDICTS = {
    "complaint_violation": {"approved", "already_punished"},
    "complaint_need_video": {"need_evidence"},
    "complaint_invalid_empty": {"invalid_format", "need_evidence"},
    "complaint_invalid_question": {"invalid_format"},
    "complaint_appeal": {"appeal_approved", "appeal_rejected"},
    "complaint_duplicate": {"duplicate"},
}
KNOWN_VERDICTS = {"approved", "already_punished", "need_evidence", "invalid_format", "appeal_approved",
                  "appeal_rejected", "duplicate", "rejected"}


def decision_agreement(results: list) -> dict:
    """Совпадает ли решение черновика с вердиктом админа (жалобы с распознанным вердиктом)."""
    n = hit = 0
    by_situation = collections.defaultdict(lambda: [0, 0])
    for row, _, m in results:
        if row["kind"] != "complaint" or row.get("verdict") not in KNOWN_VERDICTS:
            continue
        ok = row["verdict"] in DECISION_VERDICTS.get(m.get("situation"), set())
        n += 1
        hit += ok
        by_situation[m.get("situation")][0] += 1
        by_situation[m.get("situation")][1] += ok
    return {"n": n, "agreement": round(hit / n, 3) if n else None,
            "by_situation": {k: {"n": v[0], "agree": round(v[1] / v[0], 2)} for k, v in sorted(by_situation.items())}}


def duplicate_stats(results: list) -> dict:
    """Помеченные дубли против вердикта админа «дубликат» (он есть только у жалоб)."""
    flagged = [(row, m) for row, _, m in results if m.get("duplicate_of")]
    comp = [row for row, m in flagged if row["kind"] == "complaint"]
    verdict_dup = [row for row, _, _ in results if row.get("verdict") == "duplicate"]
    caught = sum(1 for row, _, m in results if row.get("verdict") == "duplicate" and m.get("duplicate_of"))
    return {
        "flagged": len(flagged),
        "flagged_reports": len(flagged) - len(comp),
        "flagged_complaints": len(comp),
        "flagged_complaint_verdicts": dict(collections.Counter(r.get("verdict") for r in comp).most_common()),
        "admin_duplicate_verdicts": len(verdict_dup),
        "admin_duplicates_caught": caught,
    }


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=config.DATA_DIR / "tickets_valid.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stage", default="stage1")
    ap.add_argument("--fake-llm", choices=["echo"], help="сухой прогон без API: модель возвращает шаблон")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if args.fake_llm == "echo":
        llm.set_drafter(llm.Drafter(client=EchoTemplateClient()))

    rules, tax = get_index(), get_taxonomy()
    rows = list(read_jsonl(args.input))
    if args.limit:
        rows = rows[: args.limit]

    workers = args.workers or (config.WORKERS if llm.enabled() else 1)
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        dups = find_duplicates(rows)
        processed = list(pool.map(lambda pair: process(pair[0], debug=True, dup=pair[1]), zip(rows, dups)))
    wall = time.perf_counter() - t_start
    results, latencies = [], []
    for row, (record, latency, meta) in zip(rows, processed):
        if "error" in record:
            raise RuntimeError(f"{row.get('ticket_id')}: {record['error']}")
        r = {k: v for k, v in record.items() if k not in ("ticket_id", "debug")}
        results.append((row, r, meta))
        latencies.append(latency)

    n = len(results)
    counters = {k: collections.Counter() for k in ("category", "priority", "lang", "situation")}
    safety = collections.Counter()
    proxy_total = proxy_hit = 0
    proxy_by_class: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    confusion = collections.Counter()
    by_kind_human = collections.defaultdict(lambda: [0, 0])

    for row, r, meta in results:
        counters["category"][r["category"]] += 1
        counters["priority"][r["priority"]] += 1
        counters["lang"][r["draft_reply"]["lang"]] += 1
        counters["situation"][meta["situation"]] += 1
        by_kind_human[row["kind"]][0] += 1
        by_kind_human[row["kind"]][1] += r["needs_human"]
        cited = r["draft_reply"]["cited"]
        safety["invalid_citations"] += sum(not rules.exists(c) for c in cited)
        safety["text_refs_not_cited"] += sum(p not in cited for p in rules.parse_refs(r["draft_reply"]["text"]))
        safety["policy_problems"] += len(meta["policy_problems"])
        if FORBIDDEN_PROMISES.search(r["draft_reply"]["text"]) and not r["needs_human"]:
            safety["promise_without_human"] += 1
        if r["category"] in tax.needs_human_always and not r["needs_human"]:
            safety["always_human_violated"] += 1
        if not cited and r["category"] not in ("report.offtopic", "report.question", "report.payment", "report.help"):
            safety["no_citation_where_rules_needed"] += 1
        gold = proxy_category(row, rules, tax)
        if gold:
            proxy_total += 1
            ok = gold == r["category"]
            proxy_hit += ok
            proxy_by_class[gold][0] += 1
            proxy_by_class[gold][1] += ok
            if not ok:
                confusion[(gold, r["category"])] += 1

    sources = collections.Counter(m["draft_source"] for _, _, m in results)
    fallbacks = collections.Counter(m["llm_fallback_reason"] for _, _, m in results if m["llm_fallback_reason"])
    cost = sum(m["llm_cost_usd"] or 0 for _, _, m in results)
    llm_lat = [m["llm_latency_ms"] for _, _, m in results if m["llm_latency_ms"] is not None]
    summary = {
        "stage": args.stage,
        "tickets": n,
        "llm_model": config.LLM_MODEL if llm.enabled() else None,
        "fake_llm": args.fake_llm,
        "latency_ms": {"p50": round(1000 * percentile(latencies, 0.5), 2), "p95": round(1000 * percentile(latencies, 0.95), 2),
                       "max": round(1000 * max(latencies), 2)},
        "llm_latency_ms": {"p50": percentile(llm_lat, 0.5), "p95": percentile(llm_lat, 0.95)} if llm_lat else None,
        "wall_s": round(wall, 1),
        "cost_usd_total": round(cost, 5),
        "cost_usd_per_ticket": round(cost / n, 6),
        "cost_usd_per_llm_draft": round(cost / sources["llm"], 6) if sources["llm"] else None,
        "draft_source": dict(sources),
        "duplicates": duplicate_stats(results),
        "decision_agreement": decision_agreement(results),
        "llm_fallback_reasons": dict(fallbacks.most_common()),
        "needs_human_share": round(sum(r["needs_human"] for _, r, _ in results) / n, 3),
        "needs_human_by_kind": {k: round(v[1] / v[0], 3) for k, v in by_kind_human.items()},
        "spam_share": round(sum(r["is_spam_or_duplicate"] for _, r, _ in results) / n, 3),
        "safety": dict(safety),
        "proxy_category": {
            "covered": proxy_total,
            "accuracy": round(proxy_hit / proxy_total, 3) if proxy_total else None,
            "by_class": {k: {"n": v[0], "acc": round(v[1] / v[0], 3)} for k, v in sorted(proxy_by_class.items())},
            "top_confusions": [[g, p, c] for (g, p), c in confusion.most_common(10)],
        },
        **{k: dict(v.most_common()) for k, v in counters.items()},
    }
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / f"eval_{args.stage}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, results, config.OUT_DIR / f"report_{args.stage}.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def write_report(s: dict, results: list, path: Path) -> None:
    lines = [
        f"# Support Copilot — отчёт ({s['stage']})",
        "",
        f"Обращений: {s['tickets']}. Модель: {s['llm_model'] or 'нет (шаблоны)'}"
        f"{' (фейковая: ' + s['fake_llm'] + ')' if s['fake_llm'] else ''}. "
        f"Стоимость: ${s['cost_usd_per_ticket']} за обращение, ${s['cost_usd_total']} всего. "
        f"Латентность p50 {s['latency_ms']['p50']} мс, p95 {s['latency_ms']['p95']} мс.",
        "",
        f"Источник черновика: {s['draft_source']}. Причины отказа от LLM: {s['llm_fallback_reasons']}.",
        "",
        f"Дубли: {s['duplicates']}.",
        "",
        f"Совпадение решения черновика с вердиктом админа: {s['decision_agreement']['agreement']} "
        f"(жалоб с распознанным вердиктом: {s['decision_agreement']['n']}).",
        "",
        "## Безопасность",
        "",
        "| Проверка | Значение |",
        "|---|---|",
    ]
    for k in ("invalid_citations", "text_refs_not_cited", "promise_without_human", "always_human_violated",
              "policy_problems", "no_citation_where_rules_needed"):
        lines.append(f"| {k} | {s['safety'].get(k, 0)} |")
    pc = s["proxy_category"]
    lines += [
        "",
        "## Категории (прокси-метки из ответа админа)",
        "",
        f"Покрыто {pc['covered']} из {s['tickets']}, accuracy {pc['accuracy']}. "
        "Прокси-метка — это решение админа, а не настоящая разметка; честная оценка будет на ручном эталоне.",
        "",
        "| Класс | n | accuracy |",
        "|---|---|---|",
    ]
    lines += [f"| {k} | {v['n']} | {v['acc']} |" for k, v in pc["by_class"].items()]
    lines += ["", "Частые ошибки (метка → наш ответ): " + "; ".join(f"{g} → {p}: {c}" for g, p, c in pc["top_confusions"]), ""]
    lines += [
        "## Распределения",
        "",
        f"Доля «нужен человек»: {pct(s['needs_human_share'])} (по типам: {s['needs_human_by_kind']}). "
        f"Спам/дубли: {pct(s['spam_share'])}.",
        "",
        f"Категории: {s['category']}",
        "",
        f"Приоритеты: {s['priority']}",
        "",
        f"Язык черновика: {s['lang']}",
        "",
        "## Примеры черновиков рядом с ответом админа",
        "",
    ]
    picked, seen = [], collections.Counter()
    for row, r, meta in results:
        key = (row["kind"], r["draft_reply"]["lang"], meta["situation"])
        if seen[key] < 1 and len(picked) < 14:
            seen[key] += 1
            picked.append((row, r, meta))
    for row, r, meta in picked:
        player = " / ".join(m["text"] for m in row["messages"] if m["role"] == "player")[:300]
        lines += [
            f"### {row['kind']} · {r['category']} · {r['priority']} · lang={r['draft_reply']['lang']} · needs_human={r['needs_human']}",
            "",
            f"**Игрок:** {player}",
            "",
            f"**Черновик** ({meta['draft_source']}, cited={r['draft_reply']['cited']}): {r['draft_reply']['text']}",
            "",
            *([f"**Шаблон этапа 1:** {meta['template_text']}", ""] if meta.get("template_text") else []),
            f"**Админ:** {admin_text(row)[:400]}",
            "",
            f"**Причины:** {'; '.join(r['reasons'])}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
