"""Очистка текста и восстановление казахских букв из cp1251-замен.

Игра работает в cp1251, поэтому на kz-серверах казахские буквы набраны похожими:
ќ→қ, ґ→ғ, ѕ→ң, є→ә, ј→ө, ў→ұ, ї→ү. Но є, ї, і, ґ — обычные украинские буквы,
поэтому замену делаем только когда текст действительно похож на казахский.
"""

from __future__ import annotations

import re
import unicodedata

_CP1251_TO_KK = str.maketrans("ќґѕєјўїЌҐЅЄЈЎЇ", "қғңәөұүҚҒҢӘӨҰҮ")
_KK_TO_CP1251 = str.maketrans("қғңәөұүһҚҒҢӘӨҰҮҺ", "ќґѕєјўїhЌҐЅЄЈЎЇH")

# Эти символы почти не встречаются вне казахских замен
_KK_ONLY_SUBS = set("ќѕјЌЅЈ")
_KK_AMBIGUOUS_SUBS = set("ґєўїҐЄЎЇ")

# Частые казахские слова (в нормальном написании). Слова, которые часто встречаются
# и в русском тексте («бан», «машина», «бар»), сюда не входят.
KK_WORDS = {
    "мен", "сен", "сіз", "сізді", "жоқ", "керек", "үшін", "қалай", "көмек", "көмектесіңіз",
    "ойын", "ойыншы", "емес", "және", "неге", "маған", "осы", "бұл", "бір", "деп",
    "қалды", "қалдым", "шықтым", "жасады", "өлтірді", "тұрып", "жатыр", "болды",
    "рахмет", "сәлем", "сәлеметсіз", "иә", "ешкім", "ештеңе", "кім", "қайда",
    "аударылып", "кіре", "алмай", "алмаймын", "жазыңыз", "шағым", "берді", "мені", "күнге",
}


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace(" ", " ").replace("\r", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.lower())


def kazakh_score(text: str) -> int:
    """Сколько слов текста похожи на казахские (после восстановления букв)."""
    fixed = text.translate(_CP1251_TO_KK).lower()
    return sum(1 for w in _words(fixed) if w in KK_WORDS) + len(re.findall(r"[әғқңөұүһ]", fixed))


def has_cp1251_kazakh(text: str) -> bool:
    if any(c in _KK_ONLY_SUBS for c in text):
        return True
    if any(c in _KK_AMBIGUOUS_SUBS for c in text):
        fixed = text.translate(_CP1251_TO_KK).lower()
        return any(w in KK_WORDS for w in _words(fixed))
    return False


def restore_kazakh(text: str) -> str:
    return text.translate(_CP1251_TO_KK)


def to_game_encoding(text: str) -> str:
    """Казахский текст -> символы, которые отображаются в cp1251-клиенте."""
    return text.translate(_KK_TO_CP1251)


def normalize(text: str) -> tuple[str, list[str]]:
    """Возвращает очищенный текст и список применённых преобразований."""
    flags: list[str] = []
    text = clean(text)
    if has_cp1251_kazakh(text):
        text = restore_kazakh(text)
        flags.append("cp1251_kazakh_restored")
    return text, flags
