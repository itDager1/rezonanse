"""Приведение вакансии из API и проверяемые замечания о ясности условий.

Пути в замечаниях относятся к полям исходного объекта вакансии, без префикса
``vacancy``. Дополнительные замечания ИИ принимаются только с реальными цитатами.
"""

from __future__ import annotations

from html import unescape
import re
from typing import Any


TOPICS = ("должность", "обязанности", "требования", "оплата", "график", "место работы", "тип занятости")

FIELD_PATHS = {
    "должность": ("job-name", "job_name", "jobName", "position", "title", "profession"),
    "обязанности": ("duty", "duties", "responsibilities", "job_responsibilities"),
    "требования": ("requirements", "requirement", "qualification", "qualifications", "skills", "education", "experience"),
    "оплата": ("salary", "salary_min", "salary_max", "salaryMin", "salaryMax", "salary_from", "salary_to", "wage", "pay", "compensation"),
    "график": ("schedule", "work_schedule", "workSchedule", "working_hours", "workingHours", "work_hours"),
    "место работы": ("addresses", "address", "workplace", "work_place", "workPlace", "location", "workPlaceType", "workplace_address"),
    "тип занятости": ("employment", "employment_type", "employmentType", "type_of_employment", "job_type"),
}
# API also puts free text in duty/requirements/qualification/term. These
# passages may describe other conditions and must be considered as evidence.
DESCRIPTION_PATHS = ("description", "job_description", "jobDescription", "text", "additional_info", "additionalInformation", "duty", "duties", "requirements", "qualification", "requirement", "term")

MISSING = {
    "должность": ("job-name", "Не указано название должности.", "Уточните название должности."),
    "обязанности": ("duty", "Не указаны должностные обязанности.", "Уточните основные обязанности сотрудника."),
    "требования": ("requirement", "Не указаны требования к кандидату.", "Уточните необходимые навыки, опыт и образование либо укажите, что требований нет."),
    "оплата": ("salary", "Не указаны сведения о размере оплаты.", "Уточните размер оплаты или диапазон и порядок его расчёта."),
    "график": ("schedule", "Не указан график работы.", "Уточните рабочие дни, смены и продолжительность рабочего времени."),
    "место работы": ("addresses", "Не указано место выполнения работы.", "Уточните адрес рабочего места или условия удалённой работы."),
    "тип занятости": ("employment", "Не указан тип занятости.", "Уточните, является ли занятость полной, частичной или иной."),
}

DESC_RE = {
    "должность": r"(?:должность|ваканси[яию]|на позици[юи])\s*[:—-]\s*\S+",
    "обязанности": r"(?:обязанност\w*|что предстоит делать|задачи|вам предстоит|должностн\w* обязанност\w*)\s*[:—-]?\s*\S+",
    "требования": r"(?:требовани\w*|от кандидата|необходим\w* опыт|образовани\w*|квалификаци\w*)\s*[:—-]?\s*\S+",
    "оплата": r"(?:(?:зарплат\w*|заработн\w*\s+плат\w*|оплат\w*|оклад\w*|вознаграждени\w*)[^\n.!?]{0,80}(?:\d|договор|обсужда|собеседован|достойн|конкурент)|(?:достойн\w*|конкурент\w*)\s+(?:зарплат\w*|заработн\w*\s+плат\w*|оплат\w*))",
    "график": r"(?:график\w*|рабочи[ейхм]+ (?:дн\w*|смен\w*|час\w*)|сменн\w* работ\w*|пятиднев\w*|шестиднев\w*)",
    "место работы": r"(?:место работы|рабочи[йхе]* адрес|работа удал[её]нно|удал[её]нная работа|работа из дома|адрес работы)\s*[:—-]?\s*\S+",
    "тип занятости": r"(?:полная|частичная|неполная|временная|сезонная)\s+занятость|(?:полный|неполный)\s+рабочий\s+день",
}


def extract_vacancy(entry: dict) -> dict:
    """Извлечь объект вакансии из обёртки ответа API (или вернуть объект)."""
    if not isinstance(entry, dict):
        raise TypeError("Вакансия должна быть JSON-объектом")
    if "vacancy" in entry:
        if not isinstance(entry["vacancy"], dict):
            raise ValueError("Поле vacancy должно содержать JSON-объект")
        return entry["vacancy"]
    return entry


def vacancy_id(entry: dict) -> Any:
    """Вернуть ID из исходных данных; не создавать и не угадывать ID."""
    vacancy = extract_vacancy(entry)
    for name in ("id", "vacancy_id", "vacancyId"):
        if name in vacancy and vacancy[name] is not None and str(vacancy[name]).strip():
            return vacancy[name]
    return None


def _primitive_paths(value: Any, path: str) -> list[dict[str, str]]:
    """Сохранить пути, включая индексы массивов, и исходные значения."""
    result: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            result.extend(_primitive_paths(nested, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            result.extend(_primitive_paths(nested, f"{path}[{index}]"))
    elif value is not None and not isinstance(value, bool):
        text = str(value).strip()
        if text:
            result.append({"field": path, "value": text})
    return result


def _path_records(vacancy: dict, paths: tuple[str, ...]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for path in paths:
        if path in vacancy:
            result.extend(_primitive_paths(vacancy[path], path))
    return result


def ai_context(entry: dict) -> dict:
    """Вернуть данные для модели с указанием полей-источников.

    ``fields`` группирует значения по семи темам, ``description`` содержит
    свободное описание. Значения и названия полей пригодны для проверки цитат.
    """
    vacancy = extract_vacancy(entry)
    return {
        "fields": {topic: _path_records(vacancy, paths) for topic, paths in FIELD_PATHS.items()},
        "description": _path_records(vacancy, DESCRIPTION_PATHS),
    }


def _plain(text: str) -> str:
    text = re.sub(r"<(?:br\s*/?|/p|/li|/div)\s*>", "\n", text, flags=re.I)
    return unescape(re.sub(r"<[^>]*>", "", text)).strip()


def _meaningful(value: str, topic: str) -> bool:
    value = _plain(value).strip(" \t\n\r-—:.;")
    if not value or value.casefold() in {"нет данных", "не указано", "не указаны", "не указан", "не задано", "null", "none", "n/a", "отсутствует"}:
        return False
    if topic == "оплата" and re.fullmatch(r"0+(?:[.,]0+)?", value):
        return False
    return True


def _useful_fields(context: dict, topic: str) -> list[dict[str, str]]:
    records = context["fields"][topic]
    if topic == "место работы":
        # Координаты, почтовый индекс и коды не описывают место работы сами по себе.
        records = [r for r in records if not re.search(r"(?:^|\.)(?:lat|lng|lon|latitude|longitude|zip|postalCode|code|regionCode|house|building)$", r["field"], re.I)]
    if topic == "требования":
        records = [r for r in records if not re.search(r"(?:^|\.)(?:code|id)$", r["field"], re.I)]
    return [r for r in records if _meaningful(r["value"], topic)]


def _desc_for(context: dict, topic: str) -> list[dict[str, str]]:
    regex = re.compile(DESC_RE[topic], re.I | re.U)
    return [r for r in context["description"] if regex.search(_plain(r["value"]))]


def _finding(topic: str, field: str, quote: str, problem: str, recommendation: str) -> dict:
    return {"topic": topic, "field": field, "quote": quote, "problem": problem, "recommendation": recommendation}


def _specific_schedule(text: str) -> bool:
    text = _plain(text).lower()
    return bool(re.search(r"\b(?:[1-7]\s*/\s*[1-7]|\d{1,2}[:.]\d{2}\s*(?:[-–—]|до)\s*\d{1,2}[:.]\d{2}|пн\s*[-–—]\s*пт|понедельник\w*\s*[-–—]\s*пятниц\w*|(?:\d{1,2})\s*(?:час\w*|ч\b)\s*(?:в|за)\s*(?:день|смен\w*)|пятиднев\w*|шестиднев\w*|(?:перв\w*|втор\w*|утренн\w*|вечерн\w*)\s+смен\w*)", text))


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(str(value).replace(" ", "").replace(",", "."))
        return result if result > 0 else None
    except (ValueError, TypeError):
        return None


def _has_salary_amount(records: list[dict[str, str]]) -> bool:
    """Оценить наличие суммы именно оплаты, а не любого числа в описании.

    Числа в структурированных границах — сумма. В свободном тексте сумма
    должна находиться рядом с упоминанием оплаты; небольшое число без валюты
    не принимается за заработок, чтобы опыт и номер дома не стали зарплатой.
    """
    structured = {"salary_min", "salary_max", "salaryMin", "salaryMax", "salary_from", "salary_to"}
    salary_fields = set(FIELD_PATHS["оплата"])
    money = r"(?:руб(?:л(?:ь|я|ей))?\.?|₽|р\.)"
    salary_cue = r"(?:зарплат\w*|заработн\w*\s+плат\w*|оплат\w*|оклад\w*|вознаграждени\w*)"
    number_pattern = r"\d[\d\s\u00a0]*(?:[.,]\d+)?"
    for record in records:
        field, value = record["field"], _plain(record["value"])
        if field in structured and _number(value) is not None:
            return True
        if field in salary_fields:
            if _number(value) is not None or re.fullmatch(r"\s*(?:от|до)\s*\d[\d\s\u00a0]*(?:[.,]\d+)?\s*", value, re.I):
                return True
            candidates = [(m.group(), value[m.end():m.end()+20]) for m in re.finditer(number_pattern, value)]
        else:
            candidates = []
            for cue in re.finditer(salary_cue, value, flags=re.I):
                nearby = value[cue.end():cue.end()+80].split(".", 1)[0].split(";", 1)[0]
                match = re.search(number_pattern, nearby)
                if match:
                    candidates.append((match.group(), nearby[match.end():match.end()+20]))
        for raw, after in candidates:
            number = _number(raw)
            if number is not None and (number >= 1000 or re.match(r"\s*" + money, after, re.I)):
                return True
    return False


def _valid_ai_finding(item: Any, context: dict) -> bool:
    """Не допускать неподтверждённые цитаты и неизвестные поля/темы."""
    if not isinstance(item, dict) or any(not isinstance(item.get(k), str) for k in ("topic", "field", "quote", "problem", "recommendation")):
        return False
    if item["topic"] not in TOPICS or not all(item[k].strip() for k in ("field", "problem", "recommendation")):
        return False
    records = [r for rs in context["fields"].values() for r in rs] + context["description"]
    field, quote = item["field"], item["quote"]
    if quote:
        return any(field == r["field"] and quote in r["value"] for r in records)
    # Пустая цитата допустима только для отсутствующего поля по теме.
    topic = item["topic"]
    possible = FIELD_PATHS[topic]
    return field in possible and not _useful_fields(context, topic) and not _desc_for(context, topic)


def analyze_vacancy(entry: dict, ai_findings: list | None = None) -> dict:
    """Вернуть отчёт по вакансии и подтверждённым замечаниям модели."""
    vacancy = extract_vacancy(entry)
    context = ai_context(vacancy)
    findings: list[dict[str, str]] = []

    for topic in TOPICS:
        if not _useful_fields(context, topic) and not _desc_for(context, topic):
            field, problem, recommendation = MISSING[topic]
            findings.append(_finding(topic, field, "", problem, recommendation))

    # Нулевой диапазон во многих выгрузках означает отсутствие значения, а не оплату.
    salary_min_key = next((k for k in ("salary_min", "salaryMin", "salary_from") if _number(vacancy.get(k)) is not None), None)
    salary_max_key = next((k for k in ("salary_max", "salaryMax", "salary_to") if _number(vacancy.get(k)) is not None), None)
    if salary_min_key and salary_max_key and _number(vacancy[salary_min_key]) > _number(vacancy[salary_max_key]):
        findings.append(_finding("оплата", f"{salary_min_key}, {salary_max_key}",
            f"{vacancy[salary_min_key]}; {vacancy[salary_max_key]}",
            "Минимальная оплата больше максимальной: указанные границы противоречат друг другу.",
            "Уточните правильные нижнюю и верхнюю границы оплаты."))

    schedule_records = _useful_fields(context, "график") + _desc_for(context, "график")
    flexible = next((r for r in schedule_records if re.search(r"гибк\w*\s+график", _plain(r["value"]), re.I)), None)
    schedule_evidence = context["fields"]["график"] + context["description"]
    if flexible and not any(_specific_schedule(r["value"]) for r in schedule_evidence):
        findings.append(_finding("график", flexible["field"], flexible["value"],
            "Указан гибкий график, но конкретные дни, смены или часы не уточнены.",
            "Уточните возможные рабочие дни, длительность смен и часы работы."))

    pay_records = _useful_fields(context, "оплата") + _desc_for(context, "оплата")
    vague_pay = next((r for r in pay_records if re.search(r"по договор[её]нности|обсужда\w*|по результатам собеседован|достойн\w*|конкурент\w*", _plain(r["value"]), re.I)), None)
    if vague_pay and not _has_salary_amount(pay_records):
        findings.append(_finding("оплата", vague_pay["field"], vague_pay["value"],
            "Оплата обозначена без конкретной суммы или диапазона.",
            "Уточните сумму или диапазон оплаты и условия его определения."))

    seen_findings = {(f["topic"], f["field"], f["quote"]) for f in findings}
    for item in ai_findings or []:
        if _valid_ai_finding(item, context):
            clean = {k: item[k] for k in ("topic", "field", "quote", "problem", "recommendation")}
            identity = (clean["topic"], clean["field"], clean["quote"])
            if identity not in seen_findings:
                findings.append(clean)
                seen_findings.add(identity)
    url = next((vacancy[k] for k in ("vac_url", "url", "vacancy_url", "vacancyUrl") if isinstance(vacancy.get(k), str) and vacancy[k].strip()), None)
    return {"id": vacancy_id(vacancy), "url": url, "findings": findings}
