"""Обучение классификатора категории: python scripts/train_classifier.py

1. Слабая разметка train из ответов админов (copilot/weak_labels.py).
2. Отложенная выборка по заявителям (20%): одни и те же игроки не попадают и в
   обучение, и в проверку. Сравниваем правила, модель без признаков правил и модель с ними.
3. Финальная модель учится на всём train и проверяется на прокси-метках valid.
4. --feedback: решения админов из интерфейса (GET /api/admin/feedback/export) —
   человеческая разметка с весом 3. Эти обращения исключаются из проверки на valid.
Сохраняет artifacts/classifier.pkl и out/train_classifier.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.classify import classify  # noqa: E402
from copilot.extract import extract  # noqa: E402
from copilot.langid import detect  # noqa: E402
from copilot.ml import KindModel, rule_features  # noqa: E402
from copilot.normalize import normalize  # noqa: E402
from copilot.predict import read_jsonl  # noqa: E402
from copilot.rules import get_index  # noqa: E402
from copilot.taxonomy import get_taxonomy  # noqa: E402
from copilot.weak_labels import label  # noqa: E402


HUMAN_WEIGHT = 3.0  # исправление админа весомее слабой метки из шаблона ответа


def prepare(rows: list[dict], rules, tax) -> list[dict]:
    """Тот же разбор, что в pipeline: текст, факты, решение правил, признаки."""
    out = []
    for row in rows:
        player = "\n".join(m["text"] for m in row["messages"] if m["role"] == "player")
        accused = any(m["role"] == "reported_player" for m in row["messages"])
        text, _ = normalize(player)
        lang = detect(text, row.get("server_lang"))[0]
        raw = row.get("rule_refs_raw") or ""
        facts = extract(text, rules, reported_nick=row.get("reported_nick"), rule_refs_raw=raw, images=row.get("images_count") or 0)
        cls = classify(row["kind"], text, facts, rules, tax, rule_refs_raw=raw)
        human = row.get("human_category")
        out.append({
            "kind": row["kind"], "text": text, "refs": raw, "group": row.get("reporter_hash") or row.get("ticket_id"),
            "feats": rule_features(row["kind"], cls, facts, lang, accused, text),
            "rule": cls.category, "label": human or label(row, rules, tax),
            "weight": HUMAN_WEIGHT if human else 1.0, "ticket_id": row.get("ticket_id"),
        })
    return out


def fit(items: list[dict], use_rules: bool = True) -> KindModel:
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=3, max_features=150000)
    refs_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True, min_df=3, max_features=20000)
    dict_vec = DictVectorizer()
    from copilot.ml import clean_text, refs_text

    char_vec.fit([clean_text(i["text"]) for i in items])
    refs_vec.fit([refs_text(i["refs"]) for i in items])
    dict_vec.fit([i["feats"] if use_rules else {} for i in items])
    model = KindModel(char_vec, refs_vec, dict_vec, None)
    X = model.matrix([i["text"] for i in items], [i["refs"] for i in items], [i["feats"] if use_rules else {} for i in items])
    clf = LogisticRegression(C=4.0, max_iter=3000)
    clf.fit(X, [i["label"] for i in items], sample_weight=[i.get("weight", 1.0) for i in items])
    model.clf = clf
    return model


def evaluate(model: KindModel, items: list[dict], use_rules: bool = True, min_prob: float = 0.0) -> dict:
    if not items:
        return {}
    X = model.matrix([i["text"] for i in items], [i["refs"] for i in items], [i["feats"] if use_rules else {} for i in items])
    proba = model.clf.predict_proba(X)
    pred = model.clf.classes_[proba.argmax(1)]
    conf = proba.max(1)
    final = [p if c >= min_prob else i["rule"] for p, c, i in zip(pred, conf, items)]
    gold = [i["label"] for i in items]
    rule = [i["rule"] for i in items]
    by_class = collections.defaultdict(lambda: [0, 0, 0])  # n, rules ok, final ok
    for g, r, f in zip(gold, rule, final):
        by_class[g][0] += 1
        by_class[g][1] += r == g
        by_class[g][2] += f == g
    return {
        "n": len(items),
        "rules_acc": round(float(np.mean([r == g for r, g in zip(rule, gold)])), 3),
        "model_acc": round(float(np.mean([f == g for f, g in zip(final, gold)])), 3),
        "by_class": {k: {"n": v[0], "rules": round(v[1] / v[0], 3), "model": round(v[2] / v[0], 3)} for k, v in sorted(by_class.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=config.CLASSIFIER_PATH)
    ap.add_argument("--feedback", type=Path, help="JSONL из /api/admin/feedback/export")
    args = ap.parse_args()
    t0 = time.time()
    rules, tax = get_index(), get_taxonomy()
    train = prepare(list(read_jsonl(config.DATA_DIR / "tickets_train.jsonl.gz")), rules, tax)
    valid = prepare(list(read_jsonl(config.DATA_DIR / "tickets_valid.jsonl")), rules, tax)
    feedback = prepare(list(read_jsonl(args.feedback)), rules, tax) if args.feedback else []
    if feedback:
        seen = {i["ticket_id"] for i in feedback}
        valid = [i for i in valid if i["ticket_id"] not in seen]  # иначе проверка на valid завышена
        print(f"feedback: {len(feedback)} human labels")
    print(f"prepared in {time.time() - t0:.0f}s")

    report: dict = {}
    models: dict = {}
    for kind in ("complaint", "report"):
        items = [i for i in train if i["kind"] == kind and i["label"]]
        groups = [i["group"] for i in items]
        tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=7).split(items, groups=groups))
        tr, te = [items[k] for k in tr_idx], [items[k] for k in te_idx]
        holdout = {}
        for name, use_rules in (("text_only", False), ("text_plus_rules", True)):
            m = fit(tr, use_rules)
            holdout[name] = evaluate(m, te, use_rules)
            if use_rules:
                holdout["thresholds"] = {str(t): evaluate(m, te, True, t)["model_acc"] for t in (0.0, 0.4, 0.5, 0.6, 0.7)}
        final = fit(items + [i for i in feedback if i["kind"] == kind and i["label"]], True)
        v_items = [i for i in valid if i["kind"] == kind and i["label"]]
        report[kind] = {
            "train_labeled": len(items),
            "holdout_by_reporter": holdout,
            "valid_proxy": {str(t): evaluate(final, v_items, True, t) for t in (0.0, config.ML_MIN_PROB)},
        }
        models[kind] = {"char_vec": final.char_vec, "refs_vec": final.refs_vec, "dict_vec": final.dict_vec, "clf": final.clf}
        print(kind, json.dumps(report[kind], ensure_ascii=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"models": models}, f, protocol=pickle.HIGHEST_PROTOCOL)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "train_classifier.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f}MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
