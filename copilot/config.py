"""Пути и флаги. Всё переопределяется переменными окружения."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """KEY=VALUE из .env (он в .gitignore); переменные окружения важнее."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

# Данные лежат вне репозитория: их нельзя публиковать.
DATA_DIR = Path(
    os.getenv(
        "COPILOT_DATA_DIR",
        ROOT.parent / "grand-support-copilot-handout-2026-10-27" / "handout",
    )
)
RULES_PATH = Path(os.getenv("COPILOT_RULES", DATA_DIR / "rules.json"))
TAXONOMY_PATH = Path(os.getenv("COPILOT_TAXONOMY", DATA_DIR / "taxonomy.json"))
OUT_DIR = Path(os.getenv("COPILOT_OUT_DIR", ROOT / "out"))
# Индекс прошлых ответов админов строится из train и содержит тексты из данных — вне git.
HISTORY_INDEX_PATH = Path(os.getenv("COPILOT_HISTORY_INDEX", ROOT / "artifacts" / "history_index.pkl"))


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes"}


# Ниже этой уверенности обращение всегда уходит человеку.
MIN_AUTO_CONFIDENCE = float(os.getenv("COPILOT_MIN_AUTO_CONFIDENCE", "0.6"))

# DM, читы и мошенничество без видео -> черновик «нужна видеозапись», а не наказание (вывод из кластеров).
NEED_VIDEO = _flag("COPILOT_NEED_VIDEO", True)

# Разрешать ли ссылки на термины из раздела «RP термины» (DM, SK, ...).
# У раздела нет пунктов, поэтому по умолчанию номер не ставим — ждём ответа организаторов.
CITE_RP_TERMS = _flag("COPILOT_CITE_RP_TERMS", False)

# --- Этап 3: ML-классификатор ---
# auto — используется, если есть файл модели; 0 — только правила.
ML_MODE = os.getenv("COPILOT_ML", "auto").strip().lower()
CLASSIFIER_PATH = Path(os.getenv("COPILOT_CLASSIFIER", ROOT / "artifacts" / "classifier.pkl"))
# Модель меняет решение правил, только если уверена не меньше порога.
ML_MIN_PROB = float(os.getenv("COPILOT_ML_MIN_PROB", "0.5"))

# --- Бонус: дубли ---
# auto/1 — искать дубли (нужны reporter_hash и created_at); 0 — выключено.
DEDUP_MODE = os.getenv("COPILOT_DEDUP", "auto").strip().lower()
DEDUP_HISTORY_PATH = Path(os.getenv("COPILOT_DEDUP_HISTORY", ROOT / "artifacts" / "dedup_history.pkl"))

# --- Бонус: интерфейс админа ---
ADMIN_DB_PATH = Path(os.getenv("COPILOT_ADMIN_DB", OUT_DIR / "admin.db"))

# --- Бонус: новые шаблоны из кластеров ---
TEMPLATE_CLUSTERS_PATH = Path(os.getenv("COPILOT_TEMPLATE_CLUSTERS", ROOT / "artifacts" / "template_clusters.pkl"))
TEMPLATE_SUGGESTIONS_PATH = Path(os.getenv("COPILOT_TEMPLATE_SUGGESTIONS", OUT_DIR / "template_suggestions.json"))

# --- RAG (обязательный шаг для каждого обращения) ---
# Похожие прошлые ответы админов (база знаний). Для массовых прогонов по train можно выключить ради скорости.
RAG_EXAMPLES = _flag("COPILOT_RAG_EXAMPLES", True)
# Пункт в черновике с наказанием выбирает поиск (среди подходящих к категории), а не таблица.
RAG_CHOOSE = _flag("COPILOT_RAG_CHOOSE", True)

# --- Этап 4: LLM-черновики ---
# auto — включено, если есть ANTHROPIC_API_KEY; 0 — только шаблоны; 1 — всегда.
LLM_MODE = os.getenv("COPILOT_LLM", "auto").strip().lower()
# Haiku 5.5: ~$0.0005 за обращение при лимите кейса $0.002. Opus 5.5 вышел бы ~$0.018.
LLM_MODEL = os.getenv("COPILOT_LLM_MODEL", "claude-haiku-5-5")
LLM_EFFORT = os.getenv("COPILOT_LLM_EFFORT", "low")
# Таймаут запроса; при превышении отдаём шаблон — так p95 остаётся < 5 с.
LLM_TIMEOUT_S = float(os.getenv("COPILOT_LLM_TIMEOUT", "3.5"))
# cited — enum из допустимых пунктов прямо в JSON-схеме ответа.
LLM_STRICT_CITED = _flag("COPILOT_LLM_STRICT_CITED", True)
# Параллельные запросы при пакетном прогоне.
WORKERS = int(os.getenv("COPILOT_WORKERS", "8"))

# Казахский ответ на внутриигровой репорт: игра показывает текст в cp1251,
# поэтому можно отдавать его «игровыми» символами (ќ вместо қ).
KK_GAME_ENCODING = _flag("COPILOT_KK_GAME_ENCODING", False)
