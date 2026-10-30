"""Замер латентности живого сервиса: python scripts/load_test.py --url http://localhost:8000"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.predict import read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--input", type=Path, default=config.DATA_DIR / "tickets_valid.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = list(read_jsonl(args.input))[: args.limit or None]
    for _ in range(60):  # ждём, пока сервер поднимется
        try:
            httpx.get(f"{args.url}/health", timeout=1).raise_for_status()
            break
        except httpx.HTTPError:
            time.sleep(0.5)

    latencies, errors = [], 0
    with httpx.Client(timeout=10) as client:
        for row in rows:
            body = {k: v for k, v in row.items() if k not in ("verdict", "punishment", "violation")}
            body["messages"] = [m for m in row["messages"] if m["role"] != "admin"]
            t0 = time.perf_counter()
            r = client.post(f"{args.url}/triage", json=body)
            latencies.append(time.perf_counter() - t0)
            errors += r.status_code != 200
    latencies.sort()
    n = len(latencies)
    print(json.dumps({"requests": n, "errors": errors,
                      "p50_ms": round(1000 * latencies[n // 2], 1),
                      "p95_ms": round(1000 * latencies[int(n * 0.95) - 1], 1),
                      "max_ms": round(1000 * latencies[-1], 1)}))


if __name__ == "__main__":
    main()
