"""Заполнить очередь админа из файла: python scripts/load_queue.py [--limit 300] [--with-reference]

Прогоняет обращения через сервис (как make predict) и кладёт в out/admin.db.
--with-reference сохраняет ответ админа из выгрузки, чтобы в демо сравнить с черновиком.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config, llm  # noqa: E402
from copilot.admin_store import get_store  # noqa: E402
from copilot.pipeline import triage  # noqa: E402
from copilot.predict import find_duplicates, read_jsonl  # noqa: E402
from copilot.schemas import TriageRequest  # noqa: E402
from copilot.weak_labels import admin_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=config.DATA_DIR / "tickets_valid.jsonl")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--with-reference", action="store_true")
    args = ap.parse_args()

    rows = list(read_jsonl(args.input))
    rows = rows[: args.limit] if args.limit else rows
    dups = find_duplicates(rows)
    store = get_store()

    def load(pair):
        row, dup = pair
        req = TriageRequest.model_validate(row)
        response, meta = triage(req, dup=dup)
        request = req.model_dump()
        request["messages"] = [m for m in request["messages"] if m["role"] != "admin"]
        return store.add(request, response.model_dump(), meta, admin_text(row) if args.with_reference else None)

    workers = config.WORKERS if llm.enabled() else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ids = list(pool.map(load, zip(rows, dups)))
    print(f"в очереди: {len(ids)} обращений -> {store.path}")


if __name__ == "__main__":
    main()
