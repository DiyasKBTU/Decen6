"""Этап 1: черновики из шаблонов на 6 языках.

Шаблоны написаны по образцу реальных ответов админов. Ссылки на правила
подставляются только из индекса, поэтому выдумать пункт шаблон не может.
Тексты kk/uz нужно вычитать носителю языка (см. docs/research-log.md).
На этапе 4 эти шаблоны остаются запасным вариантом, если LLM не ответила.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .rules import RuleIndex

LANGS = ("ru", "kk", "uz", "en", "es", "pt")

# --- как назвать пункт в тексте; формат должен разбираться обратно RuleIndex.parse_refs ---
_SECTION_NAMES = {
    "": {"ru": "общих правил", "kk": "жалпы ережелер", "uz": "Umumiy qoidalar (OP)", "en": "General Rules",
         "es": "Reglas Generales", "pt": "Regras Gerais"},
    "ЖБ": {"ru": "правил подачи жалоб", "kk": "шағым беру ережелері", "uz": "Shikoyat berish qoidalari",
           "en": "Complaint Rules", "es": "Reglas de Quejas", "pt": "Regras de Denúncias"},
    "GZ": {"ru": "правила Зелёной зоны", "kk": "Жасыл аймақ ережелері", "uz": "Yashil zona qoidalari",
           "en": "Green Zone Rules", "es": "Reglas de la Zona Verde", "pt": "Regras da Zona Verde"},
}


def ref_phrase(pid: str, lang: str, rules: RuleIndex) -> str:
    point = rules.get(pid)
    prefix, num = (point.prefix, point.ref) if point else ("", pid)
    names = _SECTION_NAMES.get(prefix)
    name = names[lang] if names else (point.section_name if point else "")
    if lang == "ru":
        if prefix == "":
            return f"п. {num} общих правил"
        if prefix == "ЖБ":
            return f"п. {num} правил подачи жалоб"
        return f"п. {num} {prefix} ({name})"
    code = f"{prefix} {num}".strip()
    return {
        "kk": f"{name}, {code}-тармақ",
        "uz": f"{name}, {code}-band",
        "en": f"{name}, rule {code}",
        "es": f"{name}, regla {code}",
        "pt": f"{name}, regra {code}",
    }[lang]


# --- суть нарушения по пунктам и терминам ---
VIOLATION = {
    "2.1": {"ru": "оскорбление игрока или его родственников", "kk": "ойыншыны немесе оның туыстарын қорлау",
            "uz": "o‘yinchini yoki uning yaqinlarini haqorat qilish", "en": "insulting a player or their relatives",
            "es": "insultar a un jugador o a sus familiares", "pt": "insultar um jogador ou seus familiares"},
    "2.2": {"ru": "пропаганда запрещённых тем", "kk": "тыйым салынған тақырыптарды насихаттау",
            "uz": "taqiqlangan mavzularni targ‘ib qilish", "en": "promoting prohibited topics",
            "es": "propaganda de temas prohibidos", "pt": "propaganda de temas proibidos"},
    "2.3": {"ru": "реклама", "kk": "жарнама", "uz": "reklama", "en": "advertising", "es": "publicidad", "pt": "publicidade"},
    "2.4": {"ru": "расизм, разжигание розни или оскорбление религии",
            "kk": "нәсілшілдік, араздық қоздыру немесе дінді қорлау",
            "uz": "irqchilik, nizo qo‘zg‘atish yoki dinni haqorat qilish",
            "en": "racism, inciting hatred or insulting religion",
            "es": "racismo, incitación al odio o insulto a la religión",
            "pt": "racismo, incitação ao ódio ou insulto à religião"},
    "2.6": {"ru": "перенос конфликта в OOC, угрозы и использование личной информации",
            "kk": "қақтығысты OOC-ке ауыстыру, қорқыту және жеке ақпаратты пайдалану",
            "uz": "nizoni OOC ga o‘tkazish, tahdid va shaxsiy ma’lumotlardan foydalanish",
            "en": "taking a conflict OOC, threats or misuse of personal information",
            "es": "llevar el conflicto a OOC, amenazas o uso de información personal",
            "pt": "levar o conflito para OOC, ameaças ou uso de informações pessoais"},
    "3.2": {"ru": "хамство или обман в отношении администрации", "kk": "әкімшілікке қатысты дөрекілік немесе алдау",
            "uz": "ma’muriyatga nisbatan qo‘pollik yoki aldov", "en": "rudeness or deception towards the administration",
            "es": "groserías o engaños hacia la administración", "pt": "grosseria ou engano com a administração"},
    "4.1": {"ru": "мошенничество по отношению к другим игрокам", "kk": "басқа ойыншыларға қатысты алаяқтық",
            "uz": "boshqa o‘yinchilarga nisbatan firibgarlik", "en": "scamming other players",
            "es": "estafa a otros jugadores", "pt": "golpe contra outros jogadores"},
    "4.2": {"ru": "вымогательство игровых ценностей", "kk": "ойын құндылықтарын бопсалап талап ету",
            "uz": "o‘yin boyliklarini tovlamachilik bilan talab qilish", "en": "extortion of in-game valuables",
            "es": "extorsión de bienes del juego", "pt": "extorsão de bens do jogo"},
    "4.4": {"ru": "сделки с игровым имуществом за реальные деньги", "kk": "ойын мүлкін нақты ақшаға сату не сатып алу",
            "uz": "o‘yin mulki bilan real pulga savdo qilish", "en": "trading in-game property for real money",
            "es": "comerciar bienes del juego por dinero real", "pt": "negociar bens do jogo por dinheiro real"},
    "5.2": {"ru": "использование стороннего ПО (читов)", "kk": "бөгде бағдарламаларды (читтерді) пайдалану",
            "uz": "begona dasturlardan (chitlardan) foydalanish", "en": "using third-party software (cheats)",
            "es": "uso de software de terceros (trampas)", "pt": "uso de software de terceiros (trapaças)"},
    "5.4": {"ru": "убийство или урон у дома или на его территории", "kk": "үйдің жанында немесе аумағында өлтіру не зақым келтіру",
            "uz": "uy oldida yoki hududida o‘ldirish yoki zarar yetkazish", "en": "killing or damaging residents at a house",
            "es": "matar o dañar a residentes en una casa", "pt": "matar ou ferir moradores em uma casa"},
    "5.6": {"ru": "отыгровка унижающих или аморальных действий", "kk": "ар-намысты қорлайтын немесе әдепсіз әрекеттерді ойнау",
            "uz": "kamsituvchi yoki axloqsiz harakatlarni rol o‘ynash", "en": "role-playing humiliating or immoral actions",
            "es": "rolear acciones humillantes o inmorales", "pt": "interpretar ações humilhantes ou imorais"},
    "1.6": {"ru": "недопустимый никнейм", "kk": "рұқсат етілмеген никнейм", "uz": "ruxsat etilmagan nikneym",
            "en": "an inappropriate nickname", "es": "un apodo inapropiado", "pt": "um apelido inadequado"},
    "GZ 1.1": {"ru": "убийство или нанесение урона в Зелёной зоне", "kk": "Жасыл аймақта өлтіру немесе зақым келтіру",
               "uz": "Yashil zonada o‘ldirish yoki zarar yetkazish", "en": "killing or dealing damage in the Green Zone",
               "es": "matar o causar daño en la Zona Verde", "pt": "matar ou causar dano na Zona Verde"},
    "GZ 1.2": {"ru": "криминальные действия против игроков в Зелёной зоне", "kk": "Жасыл аймақтағы ойыншыларға қарсы қылмыстық әрекеттер",
               "uz": "Yashil zonadagi o‘yinchilarga nisbatan jinoiy harakatlar", "en": "criminal actions against players in the Green Zone",
               "es": "acciones criminales contra jugadores en la Zona Verde", "pt": "ações criminosas contra jogadores na Zona Verde"},
}
TERM_VIOLATION = {
    "DM": {"kk": "DM — RP себебінсіз өлтіру", "uz": "DM — RP sababsiz o‘ldirish", "en": "DM — killing without an RP reason",
           "es": "DM — matar sin motivo de rol", "pt": "DM — matar sem motivo de RP"},
    "DB": {"kk": "DB — көлікпен RP себебінсіз өлтіру", "uz": "DB — transportda RP sababsiz o‘ldirish",
           "en": "DB — killing from a vehicle without an RP reason", "es": "DB — matar desde un vehículo sin motivo de rol",
           "pt": "DB — matar com veículo sem motivo de RP"},
    "SK": {"kk": "SK — пайда болу орнында өлтіру", "uz": "SK — paydo bo‘lish joyida o‘ldirish", "en": "SK — killing at a spawn point",
           "es": "SK — matar en el punto de aparición", "pt": "SK — matar no ponto de spawn"},
    "TK": {"kk": "TK — өз ұйымының мүшесін өлтіру", "uz": "TK — o‘z tashkiloti a’zosini o‘ldirish",
           "en": "TK — killing a member of your own organization", "es": "TK — matar a un miembro de tu organización",
           "pt": "TK — matar um membro da sua organização"},
    "NONRP": {"kk": "NonRP — RP режимін бұзу", "uz": "NonRP — RP rejimini buzish", "en": "NonRP — breaking the RP mode",
              "es": "NonRP — romper el modo de rol", "pt": "NonRP — quebrar o modo RP"},
}
# Самое частое решение админов в train для пункта, в пределах санкций правил
TYPICAL_PUNISHMENT = {
    "2.1": [("mute", 30)], "2.2": [("ban", 3)], "2.4": [("mute", 60)], "3.2": [("mute", 60)],
    "4.1": [("ban", 7)], "5.2": [("ban", 31)], "5.6": [("jail", 30)],
    "GZ 1.1": [("jail", 60), ("warn", 0)], "GZ 1.2": [("warn", 0)],
    "DM": [("jail", 60)], "DB": [("jail", 60)], "SK": [("jail", 90)], "TK": [("jail", 30)], "NONRP": [("jail", 30)],
}


def _ru_plural(n: int, one: str, few: str, many: str) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def render_punishment(items: list[tuple[str, int]], lang: str) -> str:
    parts = []
    for kind, n in items:
        if lang == "ru":
            parts.append({
                "mute": f"мут на {n} {_ru_plural(n, 'минуту', 'минуты', 'минут')}",
                "ban": f"блокировка на {n} {_ru_plural(n, 'день', 'дня', 'дней')}",
                "jail": f"деморган на {n} {_ru_plural(n, 'минуту', 'минуты', 'минут')}",
                "warn": "предупреждение (WARN)",
            }[kind])
            continue
        parts.append({
            "kk": {"mute": f"{n} минутқа мут", "ban": f"{n} күнге бан", "jail": f"{n} минутқа деморган", "warn": "ескерту (WARN)"},
            "uz": {"mute": f"{n} daqiqaga mute", "ban": f"{n} kunga ban", "jail": f"{n} daqiqaga demorgan", "warn": "ogohlantirish (WARN)"},
            "en": {"mute": f"a {n}-minute mute", "ban": f"a {n}-day ban", "jail": f"{n} minutes in demorgan", "warn": "a warning (WARN)"},
            "es": {"mute": f"un mute de {n} minutos", "ban": f"un baneo de {n} días", "jail": f"{n} minutos de demorgan", "warn": "una advertencia (WARN)"},
            "pt": {"mute": f"um mute de {n} minutos", "ban": f"um banimento de {n} dias", "jail": f"{n} minutos de demorgan", "warn": "uma advertência (WARN)"},
        }[lang][kind])
    return " + ".join(parts)


# --- шаблоны по ситуациям; {ref} — фраза ссылки на пункт ---
T = {
    "complaint_violation": {
        "ru": "Здравствуйте! Жалоба{about} рассмотрена. По предоставленным материалам установлено нарушение: {violation}{ref_part}. {punish}Спасибо за обращение!",
        "kk": "Сәлеметсіз бе! {about}Шағымыңыз қаралды. Ұсынылған материалдар бойынша ереже бұзушылық анықталды: {violation}{ref_part}. {punish}Өтінішіңіз үшін рахмет!",
        "uz": "Assalomu alaykum! {about}Shikoyatingiz ko‘rib chiqildi. Taqdim etilgan materiallar asosida qoidabuzarlik aniqlandi: {violation}{ref_part}. {punish}Murojaatingiz uchun rahmat!",
        "en": "Hello! Your complaint{about} has been reviewed. Based on the materials provided, a violation was found: {violation}{ref_part}. {punish}Thank you for your report!",
        "es": "¡Hola! Tu queja{about} ha sido revisada. Según los materiales presentados, se detectó una infracción: {violation}{ref_part}. {punish}¡Gracias por tu reporte!",
        "pt": "Olá! Sua denúncia{about} foi analisada. Com base nos materiais enviados, foi constatada uma infração: {violation}{ref_part}. {punish}Obrigado pela denúncia!",
    },
    "about": {"ru": " на игрока {nick}", "kk": "{nick} ойыншысына қатысты ", "uz": "{nick} ustidan ",
              "en": " about {nick}", "es": " sobre {nick}", "pt": " contra {nick}"},
    "punish_nick": {"ru": "Игрок {nick} получит наказание: {p}. ", "kk": "{nick} ойыншысына жаза қолданылады: {p}. ",
                    "uz": "{nick} jazo oladi: {p}. ", "en": "{nick} will receive {p}. ",
                    "es": "{nick} recibirá {p}. ", "pt": "{nick} receberá {p}. "},
    "punish_anon": {"ru": "Нарушитель получит наказание: {p}. ", "kk": "Бұзушыға жаза қолданылады: {p}. ",
                    "uz": "Qoidabuzar jazo oladi: {p}. ", "en": "The offender will receive {p}. ",
                    "es": "El infractor recibirá {p}. ", "pt": "O infrator receberá {p}. "},
    "punish_generic": {"ru": "К нарушителю будут применены меры по правилам проекта. ",
                       "kk": "Бұзушыға ережеге сәйкес шара қолданылады. ",
                       "uz": "Qoidabuzarga qoidalarga muvofiq chora ko‘riladi. ",
                       "en": "Measures will be taken in accordance with the rules. ",
                       "es": "Se tomarán medidas de acuerdo con las reglas. ",
                       "pt": "Serão aplicadas medidas de acordo com as regras. "},
    "scam_note": {
        "ru": " Обратите внимание: администрация не восстанавливает утраченное имущество, в том числе при мошенничестве ({ref}).",
        "kk": " Назар аударыңыз: әкімшілік жоғалған мүлікті, соның ішінде алаяқтық кезінде де, қалпына келтірмейді ({ref}).",
        "uz": " E’tibor bering: ma’muriyat yo‘qotilgan mulkni, jumladan firibgarlik holatida ham, tiklamaydi ({ref}).",
        "en": " Please note: the administration does not restore lost property, including in cases of fraud ({ref}).",
        "es": " Ten en cuenta: la administración no restaura bienes perdidos, incluso en casos de estafa ({ref}).",
        "pt": " Observação: a administração não restaura bens perdidos, inclusive em casos de golpe ({ref}).",
    },
    "complaint_review": {
        "ru": "Здравствуйте! Жалоба{about} принята. Администратор проверит приложенные доказательства и примет решение по правилам проекта. Если доказательства неполные или обрезаны, жалоба может быть отклонена ({ref}).",
        "kk": "Сәлеметсіз бе! {about}Шағымыңыз қабылданды. Әкімші тіркелген дәлелдерді тексеріп, жоба ережелеріне сай шешім қабылдайды. Дәлелдер толық болмаса немесе қиылған болса, шағым қабылданбауы мүмкін ({ref}).",
        "uz": "Assalomu alaykum! {about}Shikoyatingiz qabul qilindi. Administrator ilova qilingan dalillarni tekshirib, loyiha qoidalariga muvofiq qaror qabul qiladi. Dalillar to‘liq bo‘lmasa yoki qirqilgan bo‘lsa, shikoyat rad etilishi mumkin ({ref}).",
        "en": "Hello! Your complaint{about} has been received. An administrator will check the attached evidence and decide according to the rules. If the evidence is incomplete or cropped, the complaint may be rejected ({ref}).",
        "es": "¡Hola! Tu queja{about} ha sido recibida. Un administrador revisará las pruebas adjuntas y decidirá según las reglas. Si las pruebas están incompletas o recortadas, la queja puede ser rechazada ({ref}).",
        "pt": "Olá! Sua denúncia{about} foi recebida. Um administrador vai verificar as provas anexadas e decidir de acordo com as regras. Se as provas estiverem incompletas ou cortadas, a denúncia pode ser rejeitada ({ref}).",
    },
    "complaint_appeal": {
        "ru": "Здравствуйте! Ваше обращение по поводу наказания получено и будет передано старшей администрации. Решение примут после проверки доказательств администратора. Убедитесь, что к жалобе приложен скриншот наказания ({ref}), и не создавайте повторных жалоб — это не ускорит рассмотрение.",
        "kk": "Сәлеметсіз бе! Жазаға қатысты өтінішіңіз қабылданды және аға әкімшілікке жіберіледі. Шешім әкімшінің дәлелдері тексерілгеннен кейін қабылданады. Шағымға жазаның скриншоты тіркелгеніне көз жеткізіңіз ({ref}) және қайталама шағым жазбаңыз — бұл қарауды жылдамдатпайды.",
        "uz": "Assalomu alaykum! Jazo bo‘yicha murojaatingiz qabul qilindi va katta ma’muriyatga yuboriladi. Qaror administrator dalillari tekshirilgandan so‘ng qabul qilinadi. Shikoyatga jazo skrinshoti ilova qilinganiga ishonch hosil qiling ({ref}) va takroriy shikoyat yubormang — bu ko‘rib chiqishni tezlashtirmaydi.",
        "en": "Hello! Your appeal regarding the punishment has been received and will be forwarded to senior administration. A decision will be made after the administrator's evidence is reviewed. Please make sure a screenshot of the punishment is attached ({ref}) and do not submit duplicate complaints — this will not speed up the review.",
        "es": "¡Hola! Tu apelación sobre la sanción ha sido recibida y se enviará a la administración superior. La decisión se tomará tras revisar las pruebas del administrador. Asegúrate de adjuntar una captura de la sanción ({ref}) y no envíes quejas duplicadas: no acelerará la revisión.",
        "pt": "Olá! Seu recurso sobre a punição foi recebido e será encaminhado à administração superior. A decisão será tomada após a análise das provas do administrador. Verifique se anexou uma captura de tela da punição ({ref}) e não envie denúncias duplicadas — isso não acelera a análise.",
    },
    "complaint_invalid_empty": {
        "ru": "Здравствуйте! Жалоба не может быть рассмотрена: в ней нет точного и понятного описания произошедшего ({ref}). Подайте новую жалобу: укажите ник нарушителя, опишите, что, где и когда произошло, и приложите полные доказательства без обрезки.",
        "kk": "Сәлеметсіз бе! Шағымды қарау мүмкін емес: онда болған оқиғаның нақты әрі түсінікті сипаттамасы жоқ ({ref}). Жаңа шағым жазыңыз: бұзушының никін көрсетіңіз, не, қай жерде және қашан болғанын сипаттаңыз, толық дәлелдерді қиылмаған күйде тіркеңіз.",
        "uz": "Assalomu alaykum! Shikoyatni ko‘rib chiqib bo‘lmaydi: unda voqeaning aniq va tushunarli tavsifi yo‘q ({ref}). Yangi shikoyat yuboring: qoidabuzarning nikini ko‘rsating, nima, qayerda va qachon sodir bo‘lganini yozing hamda to‘liq, qirqilmagan dalillarni ilova qiling.",
        "en": "Hello! Your complaint cannot be reviewed: it does not contain a clear description of what happened ({ref}). Please submit a new complaint with the offender's nickname, a description of what happened, where and when, and full, uncropped evidence.",
        "es": "¡Hola! Tu queja no puede ser revisada: no contiene una descripción clara de lo ocurrido ({ref}). Envía una nueva queja con el apodo del infractor, una descripción de qué pasó, dónde y cuándo, y pruebas completas sin recortar.",
        "pt": "Olá! Sua denúncia não pode ser analisada: ela não contém uma descrição clara do que aconteceu ({ref}). Envie uma nova denúncia com o apelido do infrator, a descrição do que aconteceu, onde e quando, e provas completas, sem cortes.",
    },
    "complaint_invalid_question": {
        "ru": "Здравствуйте! Раздел жалоб не предназначен для вопросов и обсуждений ({ref}). С вопросом обратитесь в Discord-сервер игры: https://discord.gg/grnd, а с технической проблемой — в техподдержку: https://support.grnd.gg/",
        "kk": "Сәлеметсіз бе! Шағымдар бөлімі сұрақтар мен талқылауларға арналмаған ({ref}). Сұрағыңызбен ойынның Discord-серверіне жазыңыз: https://discord.gg/grnd, ал техникалық мәселе бойынша техникалық қолдауға хабарласыңыз: https://support.grnd.gg/",
        "uz": "Assalomu alaykum! Shikoyatlar bo‘limi savollar va muhokamalar uchun mo‘ljallanmagan ({ref}). Savolingiz bilan o‘yinning Discord-serveriga murojaat qiling: https://discord.gg/grnd, texnik muammo bo‘yicha esa texnik yordamga yozing: https://support.grnd.gg/",
        "en": "Hello! The complaints section is not intended for questions or discussions ({ref}). Please ask on the game's Discord server: https://discord.gg/grnd, and contact technical support for technical issues: https://support.grnd.gg/",
        "es": "¡Hola! La sección de quejas no está pensada para preguntas ni discusiones ({ref}). Pregunta en el servidor de Discord del juego: https://discord.gg/grnd, y para problemas técnicos contacta con soporte: https://support.grnd.gg/",
        "pt": "Olá! A seção de denúncias não é destinada a perguntas ou discussões ({ref}). Pergunte no servidor do Discord do jogo: https://discord.gg/grnd, e para problemas técnicos fale com o suporte: https://support.grnd.gg/",
    },
    "report_offtopic": {
        "ru": "Здравствуйте! Уточните, пожалуйста, ваш вопрос или жалобу: опишите, что случилось.",
        "kk": "Сәлеметсіз бе! Сұрағыңызды немесе шағымыңызды нақтылаңыз: не болғанын сипаттап жазыңыз.",
        "uz": "Assalomu alaykum! Iltimos, savolingiz yoki shikoyatingizni aniqlashtiring: nima bo‘lganini yozing.",
        "en": "Hello! Please clarify your question or complaint: describe what happened.",
        "es": "¡Hola! Por favor, aclara tu pregunta o queja: describe qué ha pasado.",
        "pt": "Olá! Por favor, explique sua pergunta ou reclamação: descreva o que aconteceu.",
    },
    "report_offtopic_abuse": {
        "ru": "Здравствуйте! Репорт предназначен для вопросов и жалоб, оффтоп и флуд в репорт запрещены ({ref}). Опишите, что случилось, чтобы администратор смог помочь.",
        "kk": "Сәлеметсіз бе! Репорт сұрақтар мен шағымдарға арналған, репортқа оффтоп пен флуд жазуға тыйым салынған ({ref}). Әкімші көмектесе алуы үшін не болғанын жазыңыз.",
        "uz": "Assalomu alaykum! Report savollar va shikoyatlar uchun mo‘ljallangan, reportga offtop va flud yozish taqiqlangan ({ref}). Administrator yordam bera olishi uchun nima bo‘lganini yozing.",
        "en": "Hello! Reports are for questions and complaints; off-topic messages and flooding are not allowed ({ref}). Describe what happened so an administrator can help.",
        "es": "¡Hola! Los reportes son para preguntas y quejas; el offtopic y el flood están prohibidos ({ref}). Describe qué pasó para que un administrador pueda ayudarte.",
        "pt": "Olá! Os reports são para perguntas e reclamações; offtopic e flood são proibidos ({ref}). Descreva o que aconteceu para que um administrador possa ajudar.",
    },
    "report_help": {
        "ru": "Здравствуйте! Ваш запрос передан администратору, ожидайте помощи.",
        "kk": "Сәлеметсіз бе! Сұранысыңыз әкімшіге жіберілді, көмекті күтіңіз.",
        "uz": "Assalomu alaykum! So‘rovingiz administratorga yuborildi, yordamni kuting.",
        "en": "Hello! Your request has been forwarded to an administrator, please wait for help.",
        "es": "¡Hola! Tu solicitud ha sido enviada a un administrador, espera ayuda.",
        "pt": "Olá! Sua solicitação foi encaminhada a um administrador, aguarde ajuda.",
    },
    "report_help_evac": {
        "ru": "Здравствуйте! Если транспорт перевернулся или застрял, вы можете помочь себе сами: нажмите кнопку эвакуатора на спидометре.",
        "kk": "Сәлеметсіз бе! Көлік аударылып немесе тұрып қалса, өзіңіз шеше аласыз: спидометрдегі эвакуатор түймесін басыңыз.",
        "uz": "Assalomu alaykum! Agar transport ag‘darilib yoki tiqilib qolsa, o‘zingiz hal qila olasiz: spidometrdagi evakuator tugmasini bosing.",
        "en": "Hello! If your vehicle has flipped over or got stuck, you can fix it yourself: press the tow truck button on the speedometer.",
        "es": "¡Hola! Si tu vehículo se volcó o se quedó atascado, puedes resolverlo tú mismo: pulsa el botón de grúa en el velocímetro.",
        "pt": "Olá! Se o seu veículo capotou ou ficou preso, você mesmo pode resolver: aperte o botão de guincho no velocímetro.",
    },
    "report_help_tech": {
        "ru": "Здравствуйте! С технической проблемой обратитесь в техподдержку игры: https://support.grnd.gg/",
        "kk": "Сәлеметсіз бе! Техникалық мәселе бойынша ойынның техникалық қолдау қызметіне хабарласыңыз: https://support.grnd.gg/",
        "uz": "Assalomu alaykum! Texnik muammo bo‘yicha o‘yinning texnik yordam xizmatiga murojaat qiling: https://support.grnd.gg/",
        "en": "Hello! For technical issues, please contact the game's technical support: https://support.grnd.gg/",
        "es": "¡Hola! Para problemas técnicos, contacta con el soporte técnico del juego: https://support.grnd.gg/",
        "pt": "Olá! Para problemas técnicos, fale com o suporte técnico do jogo: https://support.grnd.gg/",
    },
    "report_teleport": {
        "ru": "Здравствуйте! Администрация не предоставляет телепорт по запросу: просить телепорт, валюту или имущество через репорт запрещено ({ref}).",
        "kk": "Сәлеметсіз бе! Әкімшілік сұраныс бойынша телепорт жасамайды: репорт арқылы телепорт, валюта немесе мүлік сұрауға тыйым салынған ({ref}).",
        "uz": "Assalomu alaykum! Ma’muriyat so‘rov bo‘yicha teleport qilmaydi: report orqali teleport, valyuta yoki mulk so‘rash taqiqlangan ({ref}).",
        "en": "Hello! The administration does not provide teleports on request: asking for teleports, currency or property via reports is not allowed ({ref}).",
        "es": "¡Hola! La administración no ofrece teletransporte a petición: pedir teletransporte, dinero o bienes por reporte está prohibido ({ref}).",
        "pt": "Olá! A administração não faz teleporte a pedido: pedir teleporte, dinheiro ou itens pelo report é proibido ({ref}).",
    },
    # Из анализа кластеров: на DM, читы и мошенничество только со скриншотами админы в ~60% случаев
    # просят видео и лишь в ~10% наказывают; с видео — наоборот.
    "complaint_need_video": {
        "ru": "Здравствуйте! Жалоба{about} рассмотрена. По скриншотам нельзя однозначно подтвердить нарушение — для такого типа нарушений нужна видеозапись ({ref}). Загрузите видео на youtube.com, rutube.ru, imgur.com, Яндекс Диск или Google Диск ({ref2}) и подайте жалобу заново.",
        "kk": "Сәлеметсіз бе! {about}Шағымыңыз қаралды. Скриншоттар бойынша бұзушылықты нақты растау мүмкін емес — мұндай бұзушылық үшін бейнежазба қажет ({ref}). Бейнені youtube.com, rutube.ru, imgur.com, Яндекс Диск немесе Google Диск сайтына жүктеп ({ref2}), шағымды қайта жазыңыз.",
        "uz": "Assalomu alaykum! {about}Shikoyatingiz ko‘rib chiqildi. Skrinshotlar bo‘yicha qoidabuzarlikni aniq tasdiqlab bo‘lmaydi — bunday qoidabuzarlik uchun video yozuv kerak ({ref}). Videoni youtube.com, rutube.ru, imgur.com, Yandex Disk yoki Google Diskka yuklang ({ref2}) va shikoyatni qayta yuboring.",
        "en": "Hello! Your complaint{about} has been reviewed. The screenshots do not clearly confirm the violation — this type of violation requires a video recording ({ref}). Upload the video to youtube.com, rutube.ru, imgur.com, Yandex Disk or Google Drive ({ref2}) and submit the complaint again.",
        "es": "¡Hola! Tu queja{about} ha sido revisada. Las capturas no confirman claramente la infracción: para este tipo de infracción se necesita un video ({ref}). Sube el video a youtube.com, rutube.ru, imgur.com, Yandex Disk o Google Drive ({ref2}) y vuelve a enviar la queja.",
        "pt": "Olá! Sua denúncia{about} foi analisada. As capturas de tela não confirmam claramente a infração — para esse tipo de infração é necessário um vídeo ({ref}). Envie o vídeo para youtube.com, rutube.ru, imgur.com, Yandex Disk ou Google Drive ({ref2}) e faça a denúncia novamente.",
    },
    "complaint_duplicate": {
        "ru": "Здравствуйте! Эта жалоба повторяет вашу предыдущую жалобу на того же игрока, которая ещё рассматривается. Повторные жалобы не ускоряют рассмотрение и отклоняются как дубликаты ({ref}). Дождитесь ответа по первой жалобе.",
        "kk": "Сәлеметсіз бе! Бұл шағым сол ойыншыға қатысты әлі қаралып жатқан алдыңғы шағымыңызды қайталайды. Қайталама шағымдар қарауды жылдамдатпайды және көшірме ретінде қабылданбайды ({ref}). Бірінші шағым бойынша жауапты күтіңіз.",
        "uz": "Assalomu alaykum! Bu shikoyat o‘sha o‘yinchi ustidan hali ko‘rib chiqilayotgan oldingi shikoyatingizni takrorlaydi. Takroriy shikoyatlar ko‘rib chiqishni tezlashtirmaydi va dublikat sifatida rad etiladi ({ref}). Birinchi shikoyat bo‘yicha javobni kuting.",
        "en": "Hello! This complaint repeats your previous complaint about the same player, which is still under review. Repeated complaints do not speed up the review and are rejected as duplicates ({ref}). Please wait for the answer to your first complaint.",
        "es": "¡Hola! Esta queja repite tu queja anterior sobre el mismo jugador, que todavía se está revisando. Las quejas repetidas no aceleran la revisión y se rechazan como duplicadas ({ref}). Espera la respuesta a tu primera queja.",
        "pt": "Olá! Esta denúncia repete sua denúncia anterior contra o mesmo jogador, que ainda está em análise. Denúncias repetidas não aceleram a análise e são rejeitadas como duplicadas ({ref}). Aguarde a resposta da primeira denúncia.",
    },
    "report_duplicate": {
        "ru": "Здравствуйте! Ваше обращение уже передано администратору, повторно писать не нужно — это не ускорит ответ. Флуд в репорт запрещён ({ref}).",
        "kk": "Сәлеметсіз бе! Өтінішіңіз әкімшіге жіберіліп қойған, қайта жазудың қажеті жоқ — бұл жауапты жылдамдатпайды. Репортқа флуд жазуға тыйым салынған ({ref}).",
        "uz": "Assalomu alaykum! Murojaatingiz administratorga allaqachon yuborilgan, qayta yozish shart emas — bu javobni tezlashtirmaydi. Reportga flud yozish taqiqlangan ({ref}).",
        "en": "Hello! Your request has already been forwarded to an administrator, there is no need to write again — it will not speed up the answer. Flooding reports is not allowed ({ref}).",
        "es": "¡Hola! Tu solicitud ya fue enviada a un administrador, no hace falta escribir de nuevo: no acelerará la respuesta. El flood en reportes está prohibido ({ref}).",
        "pt": "Olá! Sua solicitação já foi encaminhada a um administrador, não é preciso escrever de novo — isso não acelera a resposta. Flood no report é proibido ({ref}).",
    },
    "report_event_teleport": {
        "ru": "Здравствуйте! Запрос на телепорт на мероприятие получен. Если телепорт открыт — ожидайте, администратор телепортирует участников; если уже закрыт, телепорт по запросу не предоставляется.",
        "kk": "Сәлеметсіз бе! Іс-шараға телепорт сұрауыңыз қабылданды. Телепорт ашық болса — күтіңіз, әкімші қатысушыларды телепорттайды; жабылып қалса, сұраныс бойынша телепорт жасалмайды.",
        "uz": "Assalomu alaykum! Tadbirga teleport so‘rovingiz qabul qilindi. Teleport ochiq bo‘lsa — kuting, administrator ishtirokchilarni teleport qiladi; yopilgan bo‘lsa, so‘rov bo‘yicha teleport qilinmaydi.",
        "en": "Hello! Your request for a teleport to the event has been received. If the teleport is open, please wait for the administrator; if it is already closed, teleports are not provided on request.",
        "es": "¡Hola! Hemos recibido tu solicitud de teletransporte al evento. Si el teletransporte está abierto, espera al administrador; si ya está cerrado, no se ofrece a petición.",
        "pt": "Olá! Recebemos seu pedido de teleporte para o evento. Se o teleporte estiver aberto, aguarde o administrador; se já estiver fechado, não é feito a pedido.",
    },
    "report_complaint": {
        "ru": "Здравствуйте! Сообщение передано администратору для проверки ситуации. Если у вас есть доказательства нарушения, подайте жалобу через лаунчер: жалоба рассматривается только при наличии объективных доказательств ({ref}).",
        "kk": "Сәлеметсіз бе! Хабарламаңыз жағдайды тексеру үшін әкімшіге жіберілді. Бұзушылыққа дәлеліңіз болса, лаунчер арқылы шағым жазыңыз: шағым тек объективті дәлелдер болғанда қаралады ({ref}).",
        "uz": "Assalomu alaykum! Xabaringiz vaziyatni tekshirish uchun administratorga yuborildi. Qoidabuzarlik bo‘yicha dalillaringiz bo‘lsa, launcher orqali shikoyat yozing: shikoyat faqat obyektiv dalillar bo‘lganda ko‘rib chiqiladi ({ref}).",
        "en": "Hello! Your message has been forwarded to an administrator to check the situation. If you have evidence of a violation, file a complaint via the launcher: complaints are reviewed only with objective evidence ({ref}).",
        "es": "¡Hola! Tu mensaje ha sido enviado a un administrador para revisar la situación. Si tienes pruebas de una infracción, presenta una queja desde el launcher: solo se revisan quejas con pruebas objetivas ({ref}).",
        "pt": "Olá! Sua mensagem foi encaminhada a um administrador para verificar a situação. Se você tem provas de uma infração, faça uma denúncia pelo launcher: só são analisadas denúncias com provas objetivas ({ref}).",
    },
    "report_kb_complaint_howto": {
        "ru": "Здравствуйте! Жалобу можно подать через лаунчер игры, приложив доказательства. Подать её нужно не позднее 48 часов с момента нарушения или наказания ({ref}).",
        "kk": "Сәлеметсіз бе! Шағымды ойын лаунчері арқылы дәлелдерін тіркеп жазуға болады. Оны бұзушылық немесе жаза болған сәттен бастап 48 сағаттан кешіктірмей беру керек ({ref}).",
        "uz": "Assalomu alaykum! Shikoyatni o‘yin launcheri orqali dalillarni ilova qilib yuborish mumkin. Uni qoidabuzarlik yoki jazo vaqtidan boshlab 48 soatdan kechiktirmay yuborish kerak ({ref}).",
        "en": "Hello! You can file a complaint via the game launcher with evidence attached. It must be filed no later than 48 hours after the incident or punishment ({ref}).",
        "es": "¡Hola! Puedes presentar una queja desde el launcher del juego adjuntando pruebas. Debe enviarse como máximo 48 horas después del incidente o la sanción ({ref}).",
        "pt": "Olá! Você pode fazer uma denúncia pelo launcher do jogo, anexando provas. Ela deve ser enviada em até 48 horas após o incidente ou a punição ({ref}).",
    },
    "report_kb_rules": {
        "ru": "Здравствуйте! С правилами игры можно ознакомиться в лаунчере.",
        "kk": "Сәлеметсіз бе! Ойын ережелерімен лаунчерде танысуға болады.",
        "uz": "Assalomu alaykum! O‘yin qoidalari bilan launcherda tanishishingiz mumkin.",
        "en": "Hello! You can read the game rules in the launcher.",
        "es": "¡Hola! Puedes consultar las reglas del juego en el launcher.",
        "pt": "Olá! Você pode ler as regras do jogo no launcher.",
    },
    "report_question": {
        "ru": "Здравствуйте! Ваш вопрос передан администратору, ожидайте ответа.",
        "kk": "Сәлеметсіз бе! Сұрағыңыз әкімшіге жіберілді, жауапты күтіңіз.",
        "uz": "Assalomu alaykum! Savolingiz administratorga yuborildi, javobni kuting.",
        "en": "Hello! Your question has been forwarded to an administrator, please wait for an answer.",
        "es": "¡Hola! Tu pregunta ha sido enviada a un administrador, espera una respuesta.",
        "pt": "Olá! Sua pergunta foi encaminhada a um administrador, aguarde a resposta.",
    },
    "report_account": {
        "ru": "Здравствуйте! Обращение по доступу к аккаунту передано администрации. Никому не сообщайте пароль и коды подтверждения. Для восстановления доступа обратитесь в техподдержку: https://support.grnd.gg/ Решение по ситуации примет администрация; восстановление имущества не гарантируется ({ref}).",
        "kk": "Сәлеметсіз бе! Аккаунтқа кіру бойынша өтінішіңіз әкімшілікке жіберілді. Құпиясөз бен растау кодтарын ешкімге айтпаңыз. Кіруді қалпына келтіру үшін техникалық қолдауға хабарласыңыз: https://support.grnd.gg/ Жағдайыңыз бойынша шешімді әкімшілік қабылдайды; мүліктің қалпына келтірілуіне кепілдік берілмейді ({ref}).",
        "uz": "Assalomu alaykum! Akkauntga kirish bo‘yicha murojaatingiz ma’muriyatga yuborildi. Parol va tasdiqlash kodlarini hech kimga aytmang. Kirishni tiklash uchun texnik yordamga murojaat qiling: https://support.grnd.gg/ Vaziyatingiz bo‘yicha qarorni ma’muriyat qabul qiladi; mulkni tiklash kafolatlanmaydi ({ref}).",
        "en": "Hello! Your account access request has been forwarded to the administration. Never share your password or verification codes. To restore access, contact technical support: https://support.grnd.gg/ The administration will decide on your case; restoration of property is not guaranteed ({ref}).",
        "es": "¡Hola! Tu solicitud sobre el acceso a la cuenta ha sido enviada a la administración. No compartas tu contraseña ni códigos de verificación. Para recuperar el acceso, contacta con soporte técnico: https://support.grnd.gg/ La administración decidirá sobre tu caso; la restauración de bienes no está garantizada ({ref}).",
        "pt": "Olá! Sua solicitação sobre o acesso à conta foi encaminhada à administração. Nunca compartilhe sua senha ou códigos de verificação. Para recuperar o acesso, fale com o suporte técnico: https://support.grnd.gg/ A administração vai decidir sobre o seu caso; a restauração de bens não é garantida ({ref}).",
    },
    "report_payment": {
        "ru": "Здравствуйте! Обращение по платежу передано специалистам. Пожалуйста, обратитесь в техподдержку: https://support.grnd.gg/ и приложите чек или номер транзакции — так проверка пройдёт быстрее.",
        "kk": "Сәлеметсіз бе! Төлем бойынша өтінішіңіз мамандарға жіберілді. Техникалық қолдауға хабарласыңыз: https://support.grnd.gg/ және чекті немесе транзакция нөмірін тіркеңіз — сонда тексеру тезірек өтеді.",
        "uz": "Assalomu alaykum! To‘lov bo‘yicha murojaatingiz mutaxassislarga yuborildi. Iltimos, texnik yordamga murojaat qiling: https://support.grnd.gg/ va chek yoki tranzaksiya raqamini ilova qiling — shunda tekshiruv tezroq o‘tadi.",
        "en": "Hello! Your payment request has been forwarded to specialists. Please contact technical support: https://support.grnd.gg/ and attach the receipt or transaction number so it can be checked faster.",
        "es": "¡Hola! Tu solicitud sobre el pago ha sido enviada a especialistas. Contacta con soporte técnico: https://support.grnd.gg/ y adjunta el recibo o el número de transacción para agilizar la revisión.",
        "pt": "Olá! Sua solicitação sobre o pagamento foi encaminhada aos especialistas. Fale com o suporte técnico: https://support.grnd.gg/ e anexe o recibo ou o número da transação para agilizar a verificação.",
    },
    "urgent": {
        "ru": " Сообщение передано администрации в первую очередь.",
        "kk": " Хабарлама әкімшілікке бірінші кезекте жіберілді.",
        "uz": " Xabar ma’muriyatga birinchi navbatda yuborildi.",
        "en": " Your message has been forwarded to the administration as a priority.",
        "es": " Tu mensaje ha sido enviado a la administración con prioridad.",
        "pt": " Sua mensagem foi encaminhada à administração com prioridade.",
    },
    "self_harm": {
        "ru": " Если вам сейчас тяжело, пожалуйста, поговорите с близкими или обратитесь в местную службу психологической помощи.",
        "kk": " Егер қазір қиын болса, жақындарыңызбен сөйлесіңіз немесе жергілікті психологиялық көмек қызметіне хабарласыңыз.",
        "uz": " Agar hozir og‘ir bo‘lsa, iltimos, yaqinlaringiz bilan gaplashing yoki mahalliy psixologik yordam xizmatiga murojaat qiling.",
        "en": " If you are having a hard time right now, please talk to someone close to you or contact a local support line.",
        "es": " Si lo estás pasando mal, habla con alguien cercano o contacta con una línea de ayuda local.",
        "pt": " Se você está passando por um momento difícil, converse com alguém próximo ou procure um serviço de apoio local.",
    },
}

# Ситуация -> пункт для цитаты по умолчанию
DEFAULT_REFS = {
    "complaint_review": "ЖБ 1.2.2",
    "complaint_appeal": "ЖБ 1.3.1",
    "complaint_invalid_empty": "ЖБ 1.2.3",
    "complaint_invalid_question": "ЖБ 1.1.2",
    "report_offtopic_abuse": "3.1",
    "report_teleport": "3.1",
    "report_complaint": "ЖБ 1.1.1",
    "report_kb_complaint_howto": "ЖБ 1.2.6",
    "report_account": "1.4",
    "complaint_duplicate": "ЖБ 1.2.9",
    "report_duplicate": "3.1",
    "complaint_need_video": "ЖБ 1.2.2",
}
# Второй пункт для шаблонов с {ref2}
SECOND_REFS = {"complaint_need_video": "ЖБ 1.2.1"}
# Ситуации, где в черновике обещано наказание: всегда подтверждает человек
PUNISHMENT_SITUATIONS = {"complaint_violation"}


@dataclass
class Draft:
    lang: str
    text: str
    cited: list[str] = field(default_factory=list)
    promises_punishment: bool = False


def build(situation: str, lang: str, rules: RuleIndex, *, nick: str | None = None, ref: str | None = None,
          rp_term: str | None = None, category: str = "", urgent: bool = False, self_harm: bool = False) -> Draft:
    lang = lang if lang in LANGS else "ru"
    cited: list[str] = []

    def cite(pid: str | None) -> str:
        if pid and rules.exists(pid):
            if pid not in cited:
                cited.append(pid)
            return ref_phrase(pid, lang, rules)
        return ""

    if situation == "report_kb_rp_term":
        definition = rules.rp_terms.get(rp_term or "", "")
        if lang == "ru" and definition:
            text = f"Здравствуйте! {rp_term} — {definition.split(' | ')[0].strip().rstrip('.')}. Подробнее — в разделе «RP термины» правил в лаунчере."
        else:
            text = T["report_question"][lang]
        draft = Draft(lang, text, cited)
    elif situation == "complaint_violation":
        draft = _violation(lang, rules, nick, ref, rp_term, category, cite)
    else:
        template = T.get(situation, T["report_question"])[lang]
        # у шаблона «нужно видео» свои пункты (правила жалоб), пункт нарушения сюда не подставляем
        own = situation in SECOND_REFS
        ref_text = cite(DEFAULT_REFS.get(situation) if own else (ref or DEFAULT_REFS.get(situation)))
        ref2_text = cite(SECOND_REFS.get(situation)) if "{ref2}" in template else ""
        about = T["about"][lang].format(nick=nick) if nick and "{about}" in template else ""
        text = _fix_about_case(template.format(ref=ref_text, ref2=ref2_text, about=about), about)
        if situation == "complaint_need_video" and category == "complaint.scam":
            text += T["scam_note"][lang].format(ref=cite("1.4"))
        draft = Draft(lang, text, cited)

    draft.cited = cited
    if urgent:
        draft.text += T["urgent"][lang]
    if self_harm:
        draft.text += T["self_harm"][lang]
    return draft


def _fix_about_case(text: str, about: str) -> str:
    """В kk/uz «Ник ойыншысына қатысты Шағымыңыз» -> «... шағымыңыз»."""
    if about:
        for word in ("Шағымыңыз", "Shikoyatingiz"):
            text = text.replace(about + word, about + word[0].lower() + word[1:])
    return text


def _violation(lang, rules, nick, ref, rp_term, category, cite) -> Draft:
    if ref and rules.exists(ref):
        violation = VIOLATION.get(ref, {}).get(lang)
        if violation is None:
            point = rules.get(ref)
            violation = point.short.rstrip(".…").lower() if lang == "ru" else {
                "kk": "ереже бұзушылық", "uz": "qoidabuzarlik", "en": "a rule violation",
                "es": "una infracción de las reglas", "pt": "uma infração das regras"}[lang]
        ref_part = f" ({cite(ref)})"
        punish_key = ref
    else:
        term = rp_term or "DM"
        if lang == "ru":
            definition = rules.rp_terms.get(term, "").split(" | ")[0].strip().rstrip(".")
            violation = f"{term} — {definition}" if definition else term
        else:
            violation = TERM_VIOLATION.get(term, TERM_VIOLATION["DM"])[lang]
        ref_part = ""
        punish_key = term

    items = TYPICAL_PUNISHMENT.get(punish_key)
    if items and nick:
        punish = T["punish_nick"][lang].format(nick=nick, p=render_punishment(items, lang))
    elif items:
        punish = T["punish_anon"][lang].format(p=render_punishment(items, lang))
    else:
        punish = T["punish_generic"][lang]
    if category == "complaint.scam":
        punish += T["scam_note"][lang].format(ref=cite("1.4")).strip() + " "
    about = T["about"][lang].format(nick=nick) if nick else ""
    text = T["complaint_violation"][lang].format(about=about, violation=violation, ref_part=ref_part, punish=punish)
    return Draft(lang, _fix_about_case(text, about), promises_punishment=True)
