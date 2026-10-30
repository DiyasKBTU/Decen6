"""История обращений для поиска дублей: python scripts/build_dedup_history.py

Сохраняет artifacts/dedup_history.pkl: по каждому заявителю — его обращения из train
(время, тип, обвиняемый, нормализованный текст, время ответа админа). Содержит тексты
из данных кейса — в репозиторий не коммитить.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.dedup import record_from_row  # noqa: E402
from copilot.predict import read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=config.DATA_DIR / "tickets_train.jsonl.gz")
    ap.add_argument("--out", type=Path, default=config.DEDUP_HISTORY_PATH)
    args = ap.parse_args()
    t0 = time.time()
    history: dict[str, list] = {}
    for row in read_jsonl(args.input):
        rec = record_from_row(row)
        if rec:
            history.setdefault(row["reporter_hash"], []).append(rec)
    for recs in history.values():
        recs.sort()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
    n = sum(len(v) for v in history.values())
    print(f"reporters={len(history)} tickets={n} size={args.out.stat().st_size / 1e6:.1f}MB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
