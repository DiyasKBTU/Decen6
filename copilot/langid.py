"""Определение языка текста игрока.

Поле `lang` в выгрузке шумное: украинский текст (є, і, ї) часто помечен как kk.
Здесь — словари частых слов + алфавитные признаки. Украинский и русский латиницей
считаем русским: игра не локализована на украинский, админы отвечают по-русски.
"""

from __future__ import annotations

import re

from .normalize import KK_WORDS

SUPPORTED = ("ru", "kk", "uz", "en", "es", "pt")

UZ_LAT_WORDS = {
    "men", "meni", "menga", "sen", "siz", "uchun", "iltimos", "yordam", "nega", "qildi", "qilib",
    "qilyapti", "bilan", "emas", "edi", "oldi", "berdi", "bor", "yoq", "yo‘q", "nima", "qanday",
    "qayerda", "kerak", "shikoyat", "boshqa", "salom", "assalomu", "alaykum", "aleykum", "rahmat",
    "mashina", "mashinani", "o‘yin", "oyin", "uldirdi", "oldirdi", "o‘ldirdi", "sokindi", "so‘kindi",
    "qamoq", "hech", "aka", "bering", "beringlar", "qoldim", "qoldi", "chiqmayapti", "sabab",
    "besabab", "mesabab", "ozi", "o‘zi", "va", "ham", "keyin", "shu", "bu", "jazo",
    "man", "mani", "qande", "qanaqa", "togrim", "odam", "sokdi", "chora", "korila", "onadan",
    "berilar", "berizla", "butunlayga", "boya", "uesin", "kun", "kunga", "daqiqa", "ottim", "otyapti",
}
UZ_CYR_WORDS = {"учун", "илтимос", "ёрдам", "нега", "қилди", "билан", "эмас", "менга", "керак", "бериш", "ўйин"}
UK_WORDS = {
    "що", "щоб", "як", "мене", "мені", "будь", "ласка", "дуже", "вже", "можна", "можу", "допоможіть",
    "тюрми", "знав", "потрібно", "розмовляти", "дати", "мій", "моя", "його", "якщо", "тому", "бачу",
}
EN_WORDS = {
    "help", "please", "stuck", "the", "my", "me", "is", "can", "you", "why", "what", "how", "player",
    "car", "killed", "kill", "insulted", "hacked", "account", "money", "pay", "paid", "where", "i'm",
    "im", "not", "with", "and", "this", "he", "she", "it", "was", "for",
}
ES_WORDS = {
    "ayuda", "ayudame", "ayúdame", "por", "favor", "estoy", "puedo", "gracias", "hola", "jugador",
    "que", "qué", "atascado", "porque", "mató", "mato", "dinero", "cuenta", "mi", "el", "la", "los",
    "insultó", "pagué", "llegó", "señor", "usted",
}
PT_WORDS = {
    "ajuda", "ajudem", "estou", "não", "nao", "obrigado", "olá", "ola", "jogador", "você", "voce",
    "preso", "matou", "pq", "está", "esta", "conta", "dinheiro", "meu", "minha", "um", "uma", "ele",
    "paguei", "chegou", "xingou", "denúncia", "cara", "mano",
}
# Русский латиницей: «pomogite», «menya ubil»
RU_LAT_WORDS = {
    "pomogite", "pomogi", "menya", "mne", "ubil", "ubili", "zastryal", "zastrial", "skrin", "skriny",
    "ne", "pozhaluysta", "pojaluysta", "pzh", "oskorbil", "prosto", "tak", "ona", "net",
    "pochemu", "kak", "gde", "ya", "menia", "chiter", "sosi", "roditeley", "rodnyh", "nakazanie",
    "snimite", "zachem", "privet", "spasibo", "rodnyu", "rodni",
}

_RE_CYR = re.compile(r"[а-яёәғқңөұүһіїєґўјѕ]", re.I)
_RE_LAT = re.compile(r"[a-zñçãõáéíóúâêô]", re.I)
_RE_KK_LETTERS = re.compile(r"[әғқңөұүһ]", re.I)
_RE_UK_LETTERS = re.compile(r"[іїєґ]", re.I)
_RE_UZ_APOS = re.compile(r"[og][‘'ʻʼ`]", re.I)


def _words(text: str) -> list[str]:
    text = re.sub(r"<(url|digits|email|handle|nick)>", " ", text.lower())
    return re.findall(r"[^\W\d_]+(?:[‘'ʻʼ][^\W\d_]+)?", text)


def detect(text: str, server_lang: str | None = None) -> tuple[str, float, str]:
    """-> (язык текста, уверенность, язык ответа).

    Язык текста: ru | kk | uz | en | es | pt | und.
    """
    words = _words(text)
    cyr = len(_RE_CYR.findall(text))
    lat = len(_RE_LAT.findall(text))
    fallback = server_lang if server_lang in ("uz",) else "ru"
    if not words or cyr + lat < 2:
        return "und", 0.3, fallback

    wset = words
    if cyr >= lat:
        kk_words = sum(w in KK_WORDS for w in wset)
        kk_letters = len(_RE_KK_LETTERS.findall(text))
        # одна случайная «һ» в русском тексте — ещё не казахский
        kk = kk_words + 2 * kk_letters if (kk_words or kk_letters >= 2) else 0
        uz = sum(w in UZ_CYR_WORDS for w in wset) + 2 * text.lower().count("ў") * (kk == 0)
        uk = sum(w in UK_WORDS for w in wset) + len(_RE_UK_LETTERS.findall(text)) * (kk == 0)
        # «мен», «бан», «машина» есть и в русском контексте — нужен перевес
        if kk >= 2 and kk >= uz:
            return "kk", min(0.95, 0.55 + 0.1 * kk), "kk"
        if uz >= 2:
            return "uz", min(0.9, 0.5 + 0.1 * uz), "uz"
        if uk >= 2:
            return "ru", 0.6, "ru"  # украинский: отвечаем по-русски
        return "ru", 0.85 if len(words) > 1 else 0.7, "ru"

    scores = {
        "uz": sum(w in UZ_LAT_WORDS for w in wset) + 2 * len(_RE_UZ_APOS.findall(text)),
        "en": sum(w in EN_WORDS for w in wset),
        "es": sum(w in ES_WORDS for w in wset) + 2 * text.count("ñ"),
        "pt": sum(w in PT_WORDS for w in wset) + 2 * len(re.findall(r"[ãõç]", text.lower())),
        "ru": sum(w in RU_LAT_WORDS for w in wset),
    }
    if server_lang == "uz":
        scores["uz"] += 1
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        # латиница без узнаваемых слов: короткие коды вроде «DM», «soft»
        return ("uz", 0.4, "uz") if server_lang == "uz" else ("ru", 0.4, "ru")
    conf = min(0.9, 0.45 + 0.12 * scores[best])
    return best, conf, best
