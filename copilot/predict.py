"""Пакетный прогон: python -m copilot.predict --in tickets.jsonl --out pred.jsonl

Каждая строка выхода — ответ по контракту плюс ticket_id. С LLM запросы идут
параллельно (COPILOT_WORKERS), порядок строк сохраняется.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, dedup, llm
from .pipeline import triage
from .retrieval import get_history_retriever
from .schemas import TriageRequest


def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def find_duplicates(rows: list[dict]) -> list:
    """Дубли считаются по порядку времени: каждое обращение сравнивается только с более ранними."""
    dups: list = [None] * len(rows)
    if not dedup.enabled():
        return dups
    order = sorted(range(len(rows)), key=lambda i: rows[i].get("created_at") or "")
    for i in order:
        dups[i] = dedup.check_and_add(rows[i])
    return dups


def process(row: dict, debug: bool = False, dup=None) -> tuple[dict, float, dict]:
    """-> (строка результата, латентность в секундах, служебные данные)."""
    started = time.perf_counter()
    try:
        response, meta = triage(TriageRequest.model_validate(row), debug=debug, dup=dup)
        record = {"ticket_id": row.get("ticket_id"), **response.model_dump()}
        if debug:
            record["debug"] = meta
    except Exception as exc:  # одна битая строка не должна ронять весь прогон
        meta = {"error": f"{type(exc).__name__}: {exc}"}
        record = {"ticket_id": row.get("ticket_id"), "error": meta["error"]}
    return record, time.perf_counter() - started, meta


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[max(0, int(len(values) * q) - 1)]


def run(in_path: Path, out_path: Path, debug: bool = False, workers: int | None = None) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(in_path))
    get_history_retriever()  # загрузить индекс до старта потоков
    dups = find_duplicates(rows)
    workers = workers or (config.WORKERS if llm.enabled() else 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda pair: process(pair[0], debug, pair[1]), zip(rows, dups)))
    with open(out_path, "w", encoding="utf-8") as out:
        for record, _, _ in results:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    latencies = [lat for _, lat, _ in results]
    metas = [m for _, _, m in results]
    cost = sum(m.get("llm_cost_usd") or 0 for m in metas)
    return {
        "tickets": len(rows),
        "errors": sum("error" in m for m in metas),
        "llm_drafts": sum(m.get("draft_source") == "llm" for m in metas),
        "duplicates": sum(d is not None for d in dups),
        "cost_usd_total": round(cost, 5),
        "cost_usd_per_ticket": round(cost / len(rows), 6) if rows else None,
        "p50_ms": round(1000 * percentile(latencies, 0.5), 1) if rows else None,
        "p95_ms": round(1000 * percentile(latencies, 0.95), 1) if rows else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--debug", action="store_true", help="добавить служебные поля")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    stats = run(args.inp, args.out, debug=args.debug, workers=args.workers)
    print(json.dumps(stats, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
