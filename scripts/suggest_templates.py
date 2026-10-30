"""Новые шаблоны из кластеров похожих обращений: python scripts/suggest_templates.py

1. Обращения train (с ответом админа) прогоняются через сервис без LLM: что мы ответили бы.
2. Тексты игроков кластеризуются отдельно по типу и языку (TF-IDF по символам + MiniBatchKMeans).
3. В каждом кластере ищется самый типичный ответ админов (медоида — ответ, ближайший к остальным)
   и его доля: насколько одинаково админы отвечают.
4. Кандидат на новый шаблон — большой кластер с единообразным ответом админов, который не похож
   на то, что сейчас отвечает сервис.

Сохраняет artifacts/template_clusters.pkl (центры кластеров — чтобы применять принятые шаблоны)
и out/template_suggestions.json / .md. Содержит тексты из данных — вне git.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.normalize import normalize  # noqa: E402
from copilot.pipeline import triage  # noqa: E402
from copilot.predict import read_jsonl  # noqa: E402
from copilot.rules import get_index  # noqa: E402
from copilot.schemas import TriageRequest  # noqa: E402
from copilot.template_library import REPLACEABLE_SITUATIONS, cluster_text  # noqa: E402

NICK = re.compile(r"\b[A-Z][a-z]{1,15}_[A-Z][a-z]{1,15}\d{0,3}\b")
GREETING = re.compile(
    r"^\W*((приветствую|здравствуйте|добрый (день|вечер|утро)|уважаем\w+( игрок)?|hurmatli|assalomu?\s*a?l[ae]ykum|salom|"
    r"сәлеметсіз бе)[\s,.!]*(вас|вы)?[\s,.!]*(<ник>)?[\s,.!]*)+",
    re.I,
)
SIGNATURE = re.compile(r"(с уважением|hurmat bilan|закрыто|yopiq)[\s\S]*$", re.I)
GENERIC = re.compile(r"слежу|ожидайте|спешу|попробую|постараюсь|в пути|лечу|работаю по|kuzat|kuting", re.I)
CACHE = config.ROOT / "artifacts" / "triage_cache_train.pkl"


def clean_answer(text: str) -> str:
    text = NICK.sub("<ник>", normalize(text)[0])
    text = re.sub(r"#\s*<digits>|№\s*#?<digits>|<digits>", "", text)
    text = GREETING.sub("", text.strip())
    text = SIGNATURE.sub("", text)
    return re.sub(r"\s+", " ", text).strip(" ,.")[:400]


def triage_all(rows: list[dict]) -> dict[str, dict]:
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    out = {}
    for row in rows:
        resp, meta = triage(TriageRequest.model_validate(row), debug=True, dup=None)
        out[row["ticket_id"]] = {"situation": meta.get("situation") or "", "draft": resp.draft_reply.text,
                                 "lang": resp.draft_reply.lang, "category": resp.category, "needs_human": resp.needs_human}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=config.DATA_DIR / "tickets_train.jsonl.gz")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    t0 = time.time()
    rules = get_index()
    rows = [r for r in read_jsonl(args.input) if any(m["role"] == "admin" for m in r["messages"])]
    ours = triage_all(rows)
    print(f"triaged {len(ours)} in {time.time() - t0:.0f}s")

    items = []
    for r in rows:
        player = " ".join(m["text"] for m in r["messages"] if m["role"] == "player")
        answer = clean_answer(" ".join(m["text"] for m in r["messages"] if m["role"] == "admin"))
        o = ours[r["ticket_id"]]
        if len(answer) < 12 or GENERIC.search(answer[:40]) or o["lang"] not in ("ru", "uz"):
            continue
        items.append({"id": r["ticket_id"], "kind": r["kind"], "lang": o["lang"], "text": cluster_text(player, r.get("rule_refs_raw")),
                      "player": NICK.sub("<ник>", player)[:200], "answer": answer, "verdict": r.get("verdict"), **o})

    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for it in items:
        groups[(it["kind"], it["lang"])].append(it)

    models, suggestions = {}, []
    for (kind, lang), its in sorted(groups.items()):
        k = max(5, min(160, len(its) // 150))
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=3, max_features=40000)
        X = vec.fit_transform([i["text"] for i in its])
        km = MiniBatchKMeans(n_clusters=k, random_state=7, n_init=3, batch_size=4096).fit(X)
        models[f"{kind}:{lang}"] = {"vectorizer": vec, "centroids": km.cluster_centers_, "kind": kind, "lang": lang}
        ans_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
        A = ans_vec.fit_transform([i["answer"] for i in its])
        # покрытие сравниваем по словам: «уточните ваш вопрос/жалобу» и наш «Уточните, пожалуйста, ваш вопрос
        # или жалобу» — один и тот же ответ, хотя по символам похожи слабо
        word_vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w\w+\b", sublinear_tf=True, min_df=2)
        word_vec.fit([i["answer"] for i in its] + [i["draft"] for i in its[:20000]])
        terms = np.array(vec.get_feature_names_out())
        for c in range(k):
            idx = np.where(km.labels_ == c)[0]
            if len(idx) < 40:
                continue
            sample = idx if len(idx) <= 600 else np.random.RandomState(c).choice(idx, 600, replace=False)
            S = cosine_similarity(A[sample])
            close = (S >= 0.55).sum(1)  # у скольких ответов в кластере почти такой же ответ
            medoid = sample[int(close.argmax())]
            share = float(close.max() / len(sample))
            situations = collections.Counter(its[j]["situation"] for j in idx).most_common(3)
            top_situation = situations[0][0]
            drafts = [its[j]["draft"] for j in sample if its[j]["situation"] == top_situation][:200]
            ours_sim = float(np.mean(cosine_similarity(word_vec.transform([its[medoid]["answer"]]), word_vec.transform(drafts))))
            verdicts = collections.Counter(its[j]["verdict"] for j in idx if its[j]["verdict"]).most_common(3)
            centroid_terms = [t.strip() for t in terms[np.argsort(km.cluster_centers_[c])[::-1][:8]] if len(t.strip()) > 1][:5]
            answer = its[medoid]["answer"]
            refs = [p for p in rules.parse_refs(answer) if rules.exists(p)]
            examples = [its[j]["player"] for j in sample[np.argsort(-S[int(close.argmax())])][:4]]
            suggestions.append({
                "cluster": f"{kind}:{lang}:{c}", "kind": kind, "lang": lang, "size": int(len(idx)), "top_situation": top_situation,
                "answer_share": round(share, 2), "similarity_to_ours": round(ours_sim, 2),
                "score": round(len(idx) * share * (1 - ours_sim), 1),
                "admin_answer": answer, "refs": refs, "terms": centroid_terms, "examples": examples,
                "our_situations": situations, "admin_verdicts": verdicts,
            })
        print(f"{kind}/{lang}: {len(its)} tickets, {k} clusters")

    # Новые шаблоны: сервис сейчас отвечает общей фразой, админы — единообразно и иначе.
    picked = [s for s in suggestions if s["answer_share"] >= 0.25 and s["similarity_to_ours"] < 0.3
              and s["top_situation"] in REPLACEABLE_SITUATIONS]
    picked.sort(key=lambda s: -s["score"])
    # Расхождения решений: сервис предлагает наказание, а админы в этом кластере чаще отказывают или просят доказательства.
    mismatches = []
    for s in suggestions:
        if s["kind"] != "complaint" or s["top_situation"] != "complaint_violation" or not s["admin_verdicts"]:
            continue
        total = sum(v for _, v in s["admin_verdicts"])
        verdict, n = s["admin_verdicts"][0]
        if verdict in ("need_evidence", "rejected", "invalid_format") and n / total >= 0.4:
            mismatches.append({**s, "dominant_verdict": verdict})
    mismatches.sort(key=lambda s: -s["size"])
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "decision_mismatches.json").write_text(json.dumps(mismatches[:20], ensure_ascii=False, indent=2), encoding="utf-8")
    seen, final = set(), []
    for s in picked:
        key = s["admin_answer"][:80].lower()
        if key not in seen:
            seen.add(key)
            final.append(s)
    final = final[: args.top]
    for i, s in enumerate(final, 1):
        s["id"] = f"T{i:02d}"

    (config.ROOT / "artifacts").mkdir(exist_ok=True)
    with open(config.TEMPLATE_CLUSTERS_PATH, "wb") as f:
        pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.TEMPLATE_SUGGESTIONS_PATH.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(final, config.OUT_DIR / "template_suggestions.md", len(suggestions))
    for s in final[:25]:
        print(f"{s['id']} {s['cluster']} size={s['size']} share={s['answer_share']} ours_sim={s['similarity_to_ours']} "
              f"ours={s['top_situation']} | {s['examples'][0][:45]!r} -> {s['admin_answer'][:100]!r}")
    print("MISMATCHES:")
    for s in mismatches[:8]:
        print(f"  {s['cluster']} size={s['size']} verdicts={s['admin_verdicts']} terms={s['terms']} | {s['examples'][0][:50]!r}")
    print(f"done in {time.time() - t0:.0f}s, {len(final)} suggestions of {len(suggestions)} clusters")


def write_markdown(final: list[dict], path: Path, n_clusters: int) -> None:
    lines = ["# Предложения новых шаблонов", "",
             f"Кластеров проверено: {n_clusters}. Кандидат — кластер, где админы отвечают единообразно, а ответ сервиса другой.", ""]
    for s in final:
        lines += [
            f"## {s['id']} · {s['kind']} · {s['lang']} · {s['size']} обращений",
            "",
            f"- Признаки: {', '.join(s['terms'])}",
            f"- Примеры: {' | '.join(s['examples'][:3])}",
            f"- Типичный ответ админов ({int(100 * s['answer_share'])}% кластера): {s['admin_answer']}",
            f"- Пункты в ответе: {', '.join(s['refs']) or 'нет'}",
            f"- Сейчас сервис: {', '.join(f'{k} ({v})' for k, v in s['our_situations'])}; вердикты админов: "
            f"{', '.join(f'{k} ({v})' for k, v in s['admin_verdicts']) or '—'}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
