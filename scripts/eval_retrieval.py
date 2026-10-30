"""Качество поиска по правилам (RAG): python scripts/eval_retrieval.py

Эталон — пункт, который админ назвал в ответе на одобренную жалобу (кроме правил жалоб ЖБ).
Метрики:
- recall@k: эталонный пункт среди первых k найденных;
- цитата совпала: эталонный пункт есть в cited нашего черновика.
Сравниваются: старый поиск (только символы текста), новый гибридный поиск, новый поиск
с подсказкой категории. Пишет out/eval_retrieval.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.normalize import normalize  # noqa: E402
from copilot.pipeline import triage  # noqa: E402
from copilot.predict import read_jsonl  # noqa: E402
from copilot.retrieval import CharRuleRetriever, get_rule_retriever  # noqa: E402
from copilot.rules import get_index  # noqa: E402
from copilot.schemas import TriageRequest  # noqa: E402
from copilot.weak_labels import admin_text  # noqa: E402


def gold_ref(row: dict, rules) -> str | None:
    if row["kind"] != "complaint" or row.get("verdict") not in ("approved", "other", "already_punished"):
        return None
    for ref in rules.parse_refs(f"{row.get('violation') or ''}\n{admin_text(row)}"):
        if not ref.startswith("ЖБ"):
            return ref
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=config.DATA_DIR / "tickets_valid.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rules = get_index()
    old, new = CharRuleRetriever(rules), get_rule_retriever()
    rows = [r for r in read_jsonl(args.input) if gold_ref(r, rules)]
    if args.limit:
        rows = rows[: args.limit]

    ks = (1, 3, 5)
    hits = {name: {k: 0 for k in ks} for name in ("old_text_only", "new_hybrid", "new_with_category")}
    cited_hit = 0
    for row in rows:
        gold = gold_ref(row, rules)
        text = normalize(" ".join(m["text"] for m in row["messages"] if m["role"] == "player"))[0]
        query = f"{text} {row.get('rule_refs_raw') or ''}"
        resp, meta = triage(TriageRequest.model_validate(row), debug=True, dup=None)
        ranked = {
            "old_text_only": [h.id for h in old.search(query, k=5, min_score=0)],
            "new_hybrid": [h.id for h in new.search(query, k=5)],
            "new_with_category": [h["id"] for h in meta["evidence"]["rules"]],
        }
        for name, ids in ranked.items():
            for k in ks:
                hits[name][k] += gold in ids[:k]
        cited_hit += gold in resp.draft_reply.cited
    n = len(rows)
    result = {
        "complaints_with_admin_rule": n,
        "recall": {name: {f"@{k}": round(v / n, 3) for k, v in h.items()} for name, h in hits.items()},
        "draft_cites_admin_rule": round(cited_hit / n, 3),
    }
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "eval_retrieval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
