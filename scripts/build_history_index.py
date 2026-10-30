"""Индекс прошлых обращений с ответами админов: python scripts/build_history_index.py

Берёт train, оставляет содержательные ответы (без «слежу за вами»), маскирует ники
и сохраняет TF-IDF индекс в artifacts/history_index.pkl. Файл содержит тексты из
данных кейса — в репозиторий не коммитить (artifacts/ в .gitignore).
"""

from __future__ import annotations

import argparse
import pickle
import random
import re
import sys
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.langid import detect  # noqa: E402
from copilot.normalize import normalize  # noqa: E402
from copilot.predict import read_jsonl  # noqa: E402

NICK = re.compile(r"\b[A-Z][a-z]{1,15}_[A-Z][a-z]{1,15}\d{0,3}\b")
# Ответы вида «слежу за вами», «ожидайте»: обещают действие админа, как образец не годятся
GENERIC_REPORT = re.compile(
    "|".join([
        "слежу", "ожидайте", "спешу", "лечу", r"(?<!\w)иду(?!\w)", "бегу", "в пути", "попробую", "постараюсь",
        "kuzat", "kuting", "yordamga bor", "бақылау", "күтіңіз", "работаю по",
    ]),
    re.I,
)


def clean_admin(text: str) -> str:
    text = NICK.sub("<ник>", normalize(text)[0])
    text = re.sub(r"\s+", " ", text).strip()
    return text[:450]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=config.DATA_DIR / "tickets_train.jsonl.gz")
    ap.add_argument("--out", type=Path, default=config.HISTORY_INDEX_PATH)
    ap.add_argument("--max-ru", type=int, default=30000, help="сколько русских обращений оставить")
    args = ap.parse_args()

    t0 = time.time()
    records, ru = [], []
    for row in read_jsonl(args.input):
        admin = " ".join(m["text"] for m in row["messages"] if m["role"] == "admin").strip()
        player = " ".join(m["text"] for m in row["messages"] if m["role"] == "player").strip()
        if len(admin) < 20 or len(player) < 2 or row.get("verdict") == "closed":
            continue
        if row["kind"] == "report" and GENERIC_REPORT.search(normalize(admin)[0][:60]):
            continue
        text, _ = normalize(player)
        lang = detect(text, row.get("server_lang"))[2]
        rec = {"kind": row["kind"], "lang": lang, "player": NICK.sub("<ник>", text)[:300],
               "admin": clean_admin(admin), "verdict": row.get("verdict")}
        (ru if lang == "ru" else records).append(rec)
    random.Random(13).shuffle(ru)
    records += ru[: args.max_ru]

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3, max_features=120000)
    matrix = vec.fit_transform([r["player"].lower() for r in records])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"vectorizer": vec, "matrix": matrix, "records": records}, f, protocol=pickle.HIGHEST_PROTOCOL)
    langs: dict[str, int] = {}
    for r in records:
        langs[r["lang"]] = langs.get(r["lang"], 0) + 1
    print(f"records={len(records)} langs={langs} size={args.out.stat().st_size / 1e6:.1f}MB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
