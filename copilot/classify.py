"""Этап 1: категория, приоритет и спам на правилах и словарях.

Каждое решение сопровождается причинами (reasons), чтобы админ видел, почему так.
На этапе 3 этот модуль станет признаками и запасным вариантом для ML-классификатора.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config
from .extract import Facts
from .rules import RuleIndex
from .taxonomy import Taxonomy


def _rx(*parts: str) -> re.Pattern:
    return re.compile("|".join(parts), re.I | re.U)


# --- словари признаков (рус., транслит, узб., каз., англ.) ---
INSULT = _rx(
    r"\bоск", r"оскорб", r"\bмат(ом|ерится|ерил)?\b", r"матер", r"родн", r"родител", r"\bмать\b", r"мамк",
    r"семь[юя]", r"послал", r"унизил", r"бляд", r"\bсук", r"\bнах", r"хуй", r"пидор", r"\bеба", r"\bёба",
    r"дебил", r"даун", r"\bчмо", r"шлюх", r"расизм", r"рас(ист|изм)", r"нацио", r"религ", r"xingou",
    r"insult", r"haqorat", r"so[‘'ʻ]?k", r"suk(in|di)", r"onadan", r"onani", r"\bosk\b", r"toxic",
)
CHEAT = _rx(
    r"\bсофт", r"\bчит", r"cheat", r"\bsoft", r"\bаим", r"\baim", r"\bwh\b", r"\bвх\b", r"wallhack",
    r"\bфлай", r"\bfly", r"летает", r"летал", r"спидхак", r"speedhack", r"сборк", r"макрос", r"багоюз",
    r"bag ?yuz", r"багаюз", r"бессмерт", r"god ?mode", r"через стен", r"сквозь стен", r"no ?recoil",
)
DM = _rx(
    r"\bдм", r"\bdm\b", r"демщик", r"деємщик", r"\bдб\b", r"\bdb\b", r"\bск\b", r"\bsk\b", r"\bтк\b",
    r"\btk\b", r"\bрк\b", r"\brk\b", r"убил", r"убива", r"убийств", r"застрелил", r"стреля", r"сбил",
    r"задавил", r"таран", r"o[‘'ʻ]?ldir", r"uldir", r"ottim", r"otyap", r"\bkill", r"mat(ou|ó)",
    r"\bзз\b", r"\bgz\b", r"\bгз\b", r"зел[её]н\w* зон", r"без причин", r"bez prichin", r"\bnrp\b", r"\bнрп",
    r"non ?rp", r"нон ?рп", r"\bурп", r"\bmg\b", r"\bмг\b", r"\bpg\b", r"\bпг\b", r"наручник", r"kishan",
    r"похит", r"мешок", r"мешк", r"в[её]р[её]вк", r"связал", r"mesabab", r"besabab",
)
SCAM = _rx(
    r"скам", r"scam", r"обман", r"\bкинул", r"\bразвел", r"\bразвёл", r"мошен", r"не отдал", r"не вернул",
    r"забрал деньги", r"aldab", r"aldadi", r"qaytarmad", r"вымог", r"шантаж", r"\bдолг", r"golpe", r"estaf",
)
RMT_ADS = _rx(
    r"реклам", r"за реал", r"реальн\w* деньг", r"\bрубл", r"\bтенге", r"телеграм", r"telegram", r"t\.me",
    r"продае?т акк", r"продаёт акк", r"продаж\w* акк", r"другой сервер", r"друг\w* проект", r"reklama",
    r"real pul", r"discord\.gg", r"\bтг\b",
)
APPEAL = _rx(
    r"не соглас", r"несоглас", r"не согл\b", r"наказал", r"наказани", r"наказан[аыо]?\b",
    r"выдал\w* (мне )?(мут|бан|джайл|варн|деморган|jail|warn|предупр|наказ)", r"сним(и|ите)\w* (мне )?(бан|мут|наказ|варн|джайл)",
    r"разбан", r"админ\w* .{0,25}(выдал|посадил|забанил|кикнул|наказал|без причин|просто так|незаконн)",
    r"забанил\w* (меня|без)", r"за что (бан|мут|посадил|наказ)", r"незаконн", r"ошибочн", r"амнист", r"amnist",
    r"\bjazo", r"menga .{0,15}(ban|mute|jail)", r"nega .{0,15}(ban|mute|jail)", r"мені бан", r"\bunban",
    r"снят\w* наказ", r"жалоба на админ", r"посадил\w* (меня )?(без|ни за)",
    # «опру» — опровержение наказания; защитная речь от первого лица
    r"\bопр[ауы]\b", r"опровер", r"\bапел", r"пересмотр", r"за (что|шо|чё|че)\b", r"где (я|вы видите|доказательств)",
    r"кого (же )?я (убил|оскорбил|оскорблял)", r"я (даже )?не (стрелял|убивал|оскорблял|нарушал|кого|никого|ничего|был|играл|оск|материл|дм)",
    r"(нет|нету)\s+(у меня\s+)?(софт|чит)", r"у меня (нет|нету) (софт|чит)", r"без читов",
    r"(мне|меня)\s+(дали|выдали|дала|дал|забанили|посадили|замутили|кикнули|перевыдали)",
    r"(дали|выдали|дала|дал)\s+(мне\s+)?(варн|ванн|бан|мут|джайл|деморган|warn|ban|mute|jail)",
    r"сним(и|ите)\b", r"\bснять\b", r"перевыдал", r"выдают бан", r"несправедлив", r"не справедлив",
    r"(неправильно|неверно) выдан", r"выдано неверно", r"nohaq", r"noqonun", r"naqonun",
    r"(menga|manga|meni)\s+.{0,25}(mut|mute|ban|warn|jail|blok)", r"blokladi", r"yechib", r"beryabdi", r"ne sogl",
)
APPEAL_RAW = re.compile(r"опр[ауы]|апел|не ?согл|наказ|сним|снять|за что|неправил|несправед|просто так|sababsiz|xato|nohaq|ne ?sogl", re.I)
PAYMENT = _rx(
    r"донат", r"donat", r"оплат", r"оплачив", r"плат[её]ж", r"списал", r"не пришл", r"не начисл",
    r"не зачисл", r"\bгк\b", r"grand ?coin", r"гранд ?коин", r"\bчек\b", r"транзакц", r"refund",
    r"возврат (денег|средств)", r"to[‘'ʻ]?lov", r"pul (tushmadi|yechildi)", r"payment", r"\bpaid\b", r"pagu[eé]",
    r"paguei", r"купил .{0,20}(не пришл|не дали)",
)
ACCOUNT = _rx(
    r"взлом", r"\bhack", r"vzlom", r"укра(ли|л)\w* (акк|аккаунт|машин|дом|бизнес|деньги|имущ)",
    r"пропал\w* (машин|дом|бизнес|деньги|имущ|акк|вещ|авто|тачк|вс[её])", r"(машин|дом|бизнес|деньги|имущ)\w* пропал",
    r"исчез", r"не могу (зайти|войти) в акк", r"парол", r"сменил\w* (пароль|почту)", r"аккаунт", r"\bакк\b",
    r"akkaunt", r"account", r"parol", r"o[‘'ʻ]?g[‘'ʻ]?irla", r"yo[‘'ʻ]?qoldi", r"зашли на мой", r"кто-?то зашел",
)
THREAT = _rx(
    r"найду (тебя|вас)", r"вычисл(ю|им)", r"по ?айпи", r"деанон", r"доксинг", r"слил\w* (мои|мой|данные|номер|адрес)",
    r"приеду к", r"убью .{0,10}(в реал|в жизни|irl)", r"в реальной жизни", r"\bswat", r"сват",
)
SELF_HARM = _rx(r"суицид", r"покончу с собой", r"убью себя", r"самоубий", r"вскро(ю|юсь)", r"не хочу жить")
MASS_ISSUE = _rx(r"у всех", r"все игроки", r"сервер (лежит|упал|не работает)", r"массов", r"всех кикнуло", r"у всех лаг")

HELP = _rx(
    r"застрял", r"зас?трял", r"текстур", r"вытащ", r"вытян", r"переверн", r"перевер", r"в вод[еуы]", r"утонул",
    r"провалил", r"под карт", r"под землю", r"\bбаг", r"глюк", r"не могу (выйти|выбраться|двигаться|встать|сесть)",
    r"наручник", r"спасите", r"zastr", r"qoldim", r"qolib", r"chiqolmay", r"tiqil", r"\bstuck", r"\bpreso",
    r"atascad", r"аударыл", r"көмектес", r"освободите", r"выпустите", r"выпустіт", r"тюрьм",
)
SELF_SERVICE_EVAC = _rx(r"переверн", r"перевер", r"аударыл", r"(машин|тачк|авто|мопед|байк|мотик)\w* .{0,15}(застрял|упал|в вод)", r"эвакуатор")
TECH = _rx(
    r"не могу (зайти|загрузиться|войти|подключиться)", r"не заход", r"вылета", r"\bлаг", r"фриз", r"не запуска",
    r"\bошибк", r"лаунчер", r"launcher", r"не грузит", r"загрузк",
)
TELEPORT = _rx(r"\bтп\b", r"\btp\b", r"телепорт", r"\bпорт(ни|ните)?\b", r"на мп\b", r"мероприят", r"ивент", r"\bevent")
REPORT_COMPLAINT_EXTRA = _rx(
    r"меша(ет|ют)", r"нарушает", r"нарушитель", r"\d{1,3}\s*(дм|дб|ск|тк|оск|софт|чит|нрп|dm|db|sk)",
    r"(дм|дб|ск|тк|софт|чит|dm|db|sk)\s*\d{1,3}", r"\bспек", r"следите", r"проследите", r"кикни", r"забань",
    r"накажите", r"\bфлуд", r"\bспам",
)
QUESTION = _rx(
    r"^(как|где|что|чем|сколько|когда|почему|зачем|какой|какая|какие|можно|кто|куда|откуда|есть ли|подскажите|скажите)\b",
    r"\b(как|где|сколько|можно ли|что такое|что делать|подскажите)\b", r"\b(qanday|qayerda|qachon|qancha)\b",
    r"\b(how|where|what|when|can i)\b", r"\b(c[oó]mo|d[oó]nde|onde|quando|cuando)\b", r"\b(қалай|қайда|қашан)\b",
)
COMPLAINT_HOWTO = _rx(r"(как|где|куда|сколько).{0,30}(жалоб|жб|пожаловаться)", r"срок.{0,15}жалоб", r"жалоб\w* .{0,15}(срок|дней|часов)")
RP_TERM_Q = re.compile(r"что так(ое|ой)\s+([a-zа-я]{2,6})|\b(mg|pg|dm|db|sk|tk|rk|nrp|мг|пг|дм|дб|ск|тк|рк|нрп)\s*(-|—)?\s*(это|что)", re.I)
RULES_WHERE = _rx(r"(где|как).{0,25}правил")
OFFTOPIC_WORDS = _rx(
    r"^(помогите|помоги|помогите пожалуйста|помогите пж|хелп|help|сос|sos|памагите|помагите|памаги|помгите|"
    r"памогите|помощь|админ|админы|ау|алло|привет|здравствуйте|hello|hi|salom|yordam|көмек|кґмек|\+|\.|\?|ку)$"
)
ABUSIVE_TO_ADMIN = _rx(r"пидор", r"пидар", r"\bеба", r"\bёба", r"хуй", r"бляд", r"\bсук[аи]\b", r"нахуй")

_TERM_ALIASES = {
    "мг": "MG", "пг": "PG", "дм": "DM", "дб": "DB", "ск": "SK", "тк": "TK", "рк": "RK", "нрп": "NONRP",
    "nrp": "NONRP", "nonrp": "NONRP", "рп": "RP", "rp": "RP", "ic": "IC", "ooc": "OOC",
}

COMPLAINT_SIGNALS = {
    "complaint.insult": INSULT,
    "complaint.cheat": CHEAT,
    "complaint.dm": DM,
    "complaint.scam": SCAM,
    "complaint.rmt_ads": RMT_ADS,
}
# Порядок при равенстве: что серьёзнее, то и выбираем
COMPLAINT_PRECEDENCE = [
    "complaint.admin", "complaint.cheat", "complaint.rmt_ads", "complaint.scam",
    "complaint.insult", "complaint.dm", "complaint.other", "complaint.invalid",
]

BASE_PRIORITY = {
    "report.account": "p0", "report.payment": "p0",
    "complaint.cheat": "p1", "complaint.rmt_ads": "p1", "complaint.scam": "p1",
    "complaint.insult": "p2", "complaint.dm": "p2", "complaint.admin": "p2", "complaint.other": "p2",
    "report.complaint": "p2", "report.help": "p2",
    "complaint.invalid": "p3", "report.question": "p3", "report.offtopic": "p3",
}


@dataclass
class Classification:
    category: str
    priority: str
    is_spam: bool
    confidence: float
    situation: str  # какой шаблон черновика нужен
    primary_ref: str | None = None
    rp_term: str | None = None
    threat: bool = False
    self_harm: bool = False
    reasons: list[str] = field(default_factory=list)
    locked: bool = False  # правило однозначное, модель его не переопределяет


def _count(rx: re.Pattern, text: str) -> int:
    return len(rx.findall(text))


def _letters(text: str) -> int:
    return len(re.findall(r"[^\W\d_]", re.sub(r"<\w+>", "", text)))


def _pick_ref(category: str, text: str, claimed: list[str], tax: Taxonomy, rules: RuleIndex) -> tuple[str | None, str | None]:
    """Пункт для цитаты: заявленный игроком, если он подходит к категории, иначе типовой."""
    for ref in claimed:
        if tax.category_for_ref(ref, "complaint") == category:
            return ref, None
    low = text.lower()
    if category == "complaint.insult":
        if re.search(r"рас(ист|изм)|нацио|религ|ксенофоб", low):
            return "2.4", None
        if THREAT.search(low):
            return "2.6", None
        return "2.1", None
    if category == "complaint.cheat":
        return "5.2", None
    if category == "complaint.scam":
        return ("4.2", None) if re.search(r"вымог|шантаж", low) else ("4.1", None)
    if category == "complaint.rmt_ads":
        return ("2.3", None) if re.search(r"реклам|сервер|проект|discord|reklama", low) else ("4.4", None)
    if category == "complaint.dm":
        if re.search(r"\bзз\b|\bzz\b|\bgz\b|\bгз\b|зел[её]н\w* зон|yashil zona", low):
            crime = re.search(r"наручник|похит|мешок|мешк|в[её]р[её]вк|связал|kishan|ограб", low)
            return ("GZ 1.2" if crime else "GZ 1.1"), None
        if re.search(r"\bдом\b|у дома|квартир|\bhouse", low):
            return "5.4", None
        for term, rx in (("SK", r"\bск\b|\bsk\b|спавн"), ("DB", r"\bдб\b|\bdb\b|сбил|задавил|таран"),
                         ("TK", r"\bтк\b|\btk\b"), ("NONRP", r"nrp|нрп|non ?rp|урп")):
            if re.search(rx, low):
                return None, term
        return None, "DM"
    if category == "complaint.other":
        return (claimed[0] if claimed else None), None
    return None, None


def classify(kind: str, text: str, facts: Facts, rules: RuleIndex, tax: Taxonomy, rule_refs_raw: str = "") -> Classification:
    low = text.lower()
    reasons: list[str] = []
    threat = bool(THREAT.search(low))
    self_harm = bool(SELF_HARM.search(low))
    if threat:
        reasons.append("угроза в реальной жизни или раскрытие личных данных")
    if self_harm:
        reasons.append("упоминание самоповреждения — нужен человек")
    if kind == "complaint":
        c = _classify_complaint(low, facts, rules, tax, rule_refs_raw, reasons)
    else:
        c = _classify_report(low, text, facts, reasons)
    c.threat, c.self_harm = threat, self_harm
    c.priority = _priority(c, kind, facts, low)
    return c


def _classify_complaint(low: str, facts: Facts, rules: RuleIndex, tax: Taxonomy, raw: str, reasons: list[str]) -> Classification:
    raw_low = (raw or "").lower()
    scores = {cat: 2 * _count(rx, low) + _count(rx, raw_low) for cat, rx in COMPLAINT_SIGNALS.items()}
    appeal = 3 * _count(APPEAL, low) + 3 * bool(APPEAL_RAW.search(raw_low))
    scores["complaint.admin"] = appeal
    for ref in facts.rule_refs:
        cat = tax.category_for_ref(ref, "complaint")
        if cat:
            scores[cat] = scores.get(cat, 0) + 3
    claimed_unmapped = [r for r in facts.rule_refs if tax.category_for_ref(r, "complaint") is None and not r.startswith("ЖБ")]
    if claimed_unmapped:
        scores["complaint.other"] = scores.get("complaint.other", 0) + 2

    best = max(COMPLAINT_PRECEDENCE[:-1], key=lambda c: (scores.get(c, 0), -COMPLAINT_PRECEDENCE.index(c)))
    top = scores.get(best, 0)

    if top == 0:
        if _letters(low) >= 12 and not QUESTION.search(low):
            reasons.append("признаков конкретного нарушения нет, но есть описание")
            return Classification("complaint.other", "", False, 0.4, "complaint_review")
        c = finalize_complaint("complaint.invalid", low, facts, rules, tax, 0.65, reasons)
        c.locked = _letters(low) < 6  # «.», «жб», «вот он» — по таксономии это точно не жалоба
        return c
    second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
    conf = min(0.9, 0.55 + 0.1 * top) if best == "complaint.admin" else min(0.9, 0.5 + 0.08 * top + 0.05 * (top - second))
    return finalize_complaint(best, low, facts, rules, tax, conf, reasons)


def finalize_complaint(category: str, low: str, facts: Facts, rules: RuleIndex, tax: Taxonomy, conf: float,
                       reasons: list[str]) -> Classification:
    """Категория жалобы -> ситуация для черновика и пункт для цитаты."""
    if category == "complaint.invalid":
        question = bool(QUESTION.search(low))
        reasons.append("это вопрос, а не жалоба" if question else "нет описания нарушения и ссылки на правило")
        return Classification(category, "", False, conf, "complaint_invalid_question" if question else "complaint_invalid_empty")
    if category == "complaint.admin":
        reasons.append("обжалование наказания или жалоба на администратора")
        return Classification(category, "", False, conf, "complaint_appeal")
    ref, term = _pick_ref(category, low, facts.rule_refs, tax, rules)
    reasons.append(f"признаки категории {category}")
    if facts.rule_refs and ref and ref not in facts.rule_refs:
        reasons.append(f"игрок указал {', '.join(facts.rule_refs)}, по описанию подходит {ref}")
    situation = "complaint_violation" if (ref or term) else "complaint_review"
    if config.NEED_VIDEO and category in NEED_VIDEO_CATEGORIES and facts.evidence_kind != "video":
        # train: только скриншоты -> админы просят видео в ~60% случаев с известным вердиктом, наказывают в ~10%
        situation = "complaint_need_video"
        reasons.append("для такого нарушения админы обычно требуют видео, а его нет")
    return Classification(category, "", False, conf, situation, primary_ref=ref, rp_term=term)


# Нарушения, которые по скриншоту не доказать (вывод из анализа кластеров, docs/research-log.md)
NEED_VIDEO_CATEGORIES = {"complaint.dm", "complaint.cheat", "complaint.scam"}


# Классы, где правила точнее модели: модель переубеждает их только при высокой уверенности.
# DM: в слабой разметке DM-жалоба получает метку, только если её одобрили (без видео — редко),
# а апелляции размечены всегда; модель выучила «без причины -> апелляция». Правила по DM точнее
# (valid 92% против 87%, отложенная выборка 85% против 83%).
PROTECTED = {"complaint.cheat", "complaint.rmt_ads", "complaint.scam", "complaint.invalid", "complaint.dm"}
GENERIC_REPORT_SITUATIONS = {"report_complaint": "report.complaint", "report_help": "report.help", "report_offtopic": "report.offtopic"}
REPORT_SITUATION_BY_CATEGORY = {v: k for k, v in GENERIC_REPORT_SITUATIONS.items()}


def refine_with_model(c: Classification, kind: str, text: str, facts: Facts, rules: RuleIndex, tax: Taxonomy,
                      category: str, prob: float, min_prob: float, strong_prob: float = 0.8) -> Classification:
    """Решение правил + предсказание модели -> итог. Модель меняет только то, где правила не однозначны."""
    if c.locked or category == c.category or prob < min_prob:
        return c
    low = text.lower()
    if kind == "complaint":
        if c.category in PROTECTED and c.situation != "complaint_review" and prob < strong_prob:
            return c
        keep = [r for r in c.reasons if "угроз" in r or "самоповреж" in r]
        new = finalize_complaint(category, low, facts, rules, tax, round(prob, 2), keep)
    else:
        if c.situation not in GENERIC_REPORT_SITUATIONS or category not in REPORT_SITUATION_BY_CATEGORY:
            return c
        # для репортов модель учили только на трёх классах и почти без английского — порог выше;
        # «оффтоп» стирает смысл обращения, поэтому туда — только при высокой уверенности
        if prob < max(min_prob, 0.65) or (category == "report.offtopic" and prob < strong_prob):
            return c
        keep = [r for r in c.reasons if "угроз" in r or "самоповреж" in r]
        spam = category == "report.offtopic" and len(re.findall(r"[^\W\d_]+", low)) <= 2
        new = Classification(category, "", spam, round(prob, 2), REPORT_SITUATION_BY_CATEGORY[category], reasons=keep)
    new.reasons.append(f"классификатор: {category} (p={prob:.2f}), правила: {c.category}")
    new.threat, new.self_harm = c.threat, c.self_harm
    new.priority = _priority(new, kind, facts, low)
    return new


def _classify_report(low: str, text: str, facts: Facts, reasons: list[str]) -> Classification:
    stripped = low.strip(" .,!?")
    if PAYMENT.search(low):
        reasons.append("платёж или донат — всегда человек")
        return Classification("report.payment", "p0", False, 0.8, "report_payment", locked=True)
    if ACCOUNT.search(low) and not RMT_ADS.search(low):
        reasons.append("доступ к аккаунту или пропажа имущества — всегда человек")
        return Classification("report.account", "p0", False, 0.75, "report_account", locked=True)
    if re.fullmatch(r"\s*\++\s*", low):
        # так игроки отвечают на объявление о телепорте на мероприятие; админы пишут «ожидайте» или «телепорт закрыт»
        reasons.append("«+» — просьба о телепорте на мероприятие")
        return Classification("report.help", "p3", False, 0.75, "report_event_teleport", locked=True)
    if _letters(low) == 0 or OFFTOPIC_WORDS.search(stripped):
        abusive = bool(ABUSIVE_TO_ADMIN.search(low))
        reasons.append("сообщение без содержания")
        return Classification("report.offtopic", "p3", True, 0.85, "report_offtopic_abuse" if abusive else "report_offtopic",
                              locked=True)
    if COMPLAINT_HOWTO.search(low):
        reasons.append("вопрос о подаче жалобы — ответ есть в правилах жалоб")
        return Classification("report.question", "p3", False, 0.8, "report_kb_complaint_howto", primary_ref="ЖБ 1.2.6", locked=True)
    m = RP_TERM_Q.search(low)
    if m:
        term = _TERM_ALIASES.get((m.group(2) or m.group(3) or "").lower())
        if term:
            reasons.append(f"вопрос о термине {term} — ответ в разделе «RP термины»")
            return Classification("report.question", "p3", False, 0.8, "report_kb_rp_term", rp_term=term, locked=True)
    if RULES_WHERE.search(low):
        reasons.append("вопрос о правилах")
        return Classification("report.question", "p2", False, 0.75, "report_kb_rules", locked=True)
    if ABUSIVE_TO_ADMIN.search(low) and _letters(low) < 40:
        reasons.append("оскорбительный оффтоп в репорт")
        return Classification("report.offtopic", "p3", True, 0.7, "report_offtopic_abuse")

    complaint_hits = _count(DM, low) + _count(INSULT, low) + _count(CHEAT, low) + _count(REPORT_COMPLAINT_EXTRA, low)
    help_hits = _count(HELP, low)
    if SELF_SERVICE_EVAC.search(low) and complaint_hits == 0:
        reasons.append("перевёрнутый или застрявший транспорт — есть кнопка эвакуатора")
        return Classification("report.help", "p3", False, 0.8, "report_help_evac", locked=True)
    if TECH.search(low) and complaint_hits == 0:
        reasons.append("техническая проблема — нужна техподдержка")
        return Classification("report.help", "p2", False, 0.7, "report_help_tech", locked=True)
    if TELEPORT.search(low) and complaint_hits == 0 and help_hits == 0:
        reasons.append("просьба о телепорте — запрещено п. 3.1")
        return Classification("report.help", "p3", False, 0.7, "report_teleport", primary_ref="3.1", locked=True)
    if complaint_hits > help_hits:
        reasons.append("жалоба на игрока через репорт — админ проверяет в игре")
        return Classification("report.complaint", "p2", False, min(0.85, 0.55 + 0.1 * complaint_hits), "report_complaint")
    if help_hits:
        reasons.append("просьба о помощи — нужно действие админа в игре")
        return Classification("report.help", "p2", False, min(0.85, 0.55 + 0.1 * help_hits), "report_help")
    if QUESTION.search(low) or "?" in low:
        reasons.append("вопрос, готового ответа в базе нет")
        return Classification("report.question", "p3", False, 0.55, "report_question")
    words = re.findall(r"[^\W\d_]+", low)
    if len(words) <= 2:
        reasons.append("короткое сообщение без сути")
        return Classification("report.offtopic", "p3", True, 0.6, "report_offtopic")
    reasons.append("тип обращения не распознан")
    return Classification("report.question", "p3", False, 0.4, "report_question")


def _priority(c: Classification, kind: str, facts: Facts, low: str) -> str:
    if c.threat or c.self_harm:
        return "p0"
    if kind == "report" and MASS_ISSUE.search(low):
        c.reasons.append("похоже на массовую проблему")
        return "p0"
    p = c.priority or BASE_PRIORITY.get(c.category, "p2")
    if kind == "complaint" and c.category not in ("complaint.invalid",):
        if facts.evidence_kind == "video" and p == "p2" and c.category != "complaint.admin":
            c.reasons.append("есть видеодоказательство")
            p = "p1"
        elif facts.evidence_kind is None and facts.images == 0:
            c.reasons.append("нет доказательств")
            p = "p3"
    return p
