from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from vacancy_analyzer.analyzer import ai_context, analyze_vacancy, extract_vacancy, vacancy_id
from vacancy_analyzer.ai import AIAnalyzer, AIError

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = "https://opendata.trudvsem.ru/api/v1/vacancies/region/39"


def load_env(path: Path = ROOT / ".env") -> None:
    """Поддерживаем простой .env без исполнения кода и без перезаписи окружения."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and (key[0].isalpha() or key[0] == "_") and all(c.isalnum() or c == "_" for c in key):
            os.environ.setdefault(key, value.strip('"\''))


def records_from_json(payload: object) -> list[dict]:
    """Принимает список либо объект ответа API. Не маскирует ошибочный формат под пустой набор."""
    if isinstance(payload, list):
        if any(not isinstance(x, dict) for x in payload):
            raise ValueError("Список вакансий должен содержать объекты JSON")
        return payload
    if not isinstance(payload, dict):
        raise ValueError("Ожидался объект ответа API или список вакансий")
    status = payload.get("status")
    if status is not None and str(status).lower() not in ("200", "ok", "success"):
        raise ValueError(f"API сообщает об ошибке: status={status}")
    results = payload.get("results")
    if isinstance(results, dict) and isinstance(results.get("vacancies"), list):
        return records_from_json(results["vacancies"])
    if isinstance(payload.get("vacancies"), list):
        return records_from_json(payload["vacancies"])
    raise ValueError("В JSON отсутствует массив results.vacancies или vacancies")


def api_page(url: str, offset: int, limit: int, timeout: int = 30) -> list[dict]:
    parts = urlsplit(url)
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    params.update({"offset": str(offset), "limit": str(limit)})
    request_url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))
    req = Request(request_url, headers={"Accept": "application/json", "User-Agent": "ResonanceVacancyPrototype/1.0"})
    for attempt in range(2):
        try:
            with urlopen(req, timeout=timeout) as response:
                raw = response.read()
            payload = json.loads(raw.decode("utf-8-sig"))
            return records_from_json(payload)
        except (HTTPError, URLError, TimeoutError, UnicodeError, json.JSONDecodeError, OSError) as exc:
            if attempt == 0 and (not isinstance(exc, HTTPError) or exc.code >= 500):
                print(f"API временно недоступен (offset={offset}); повтор запроса.", file=sys.stderr, flush=True)
                time.sleep(1)
                continue
            raise RuntimeError(f"Не удалось получить страницу API (offset={offset}): {exc}") from exc


def fetch_vacancies(url: str, limit: int, page_size: int = 20) -> list[dict]:
    if limit < 1 or not 1 <= page_size <= 100:
        raise ValueError("Лимит должен быть больше нуля, размер страницы — от 1 до 100")
    vacancies: list[dict] = []
    seen: set[str] = set()
    offset = 0
    while len(vacancies) < limit:
        # В этом API offset — индекс страницы, а не количество пропущенных
        # записей: размер страницы должен оставаться постоянным на всех запросах.
        requested = page_size
        page = api_page(url, offset, requested)
        print(f"Получена страница: offset={offset}, записей={len(page)}", flush=True)
        if not page:
            break
        new = 0
        for item in page:
            record = extract_vacancy(item)
            ident = vacancy_id(record)
            if ident is None or str(ident).strip() == "":
                raise ValueError("В полученной вакансии отсутствует ID")
            if str(ident) not in seen:
                vacancies.append(record)
                seen.add(str(ident))
                new += 1
        if new == 0:
            raise ValueError("API повторил страницу без новых ID; отчёт не обновлён")
        offset += 1
        if len(page) < requested:
            break
    return vacancies[:limit]


def unique_records(entries: list[dict]) -> list[dict]:
    result = []
    seen = set()
    for entry in entries:
        vacancy = extract_vacancy(entry)
        ident = vacancy_id(vacancy)
        if ident is None or str(ident).strip() == "":
            raise ValueError("В вакансии отсутствует ID")
        if str(ident) not in seen:
            seen.add(str(ident))
            result.append(vacancy)
    return result


def write_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".report-", suffix=".tmp", delete=False) as file:
            temp_path = Path(file.name)
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def build_report(records: list[dict], provider: str) -> list[dict]:
    model = AIAnalyzer.from_env(provider)
    report = []
    for i, vacancy in enumerate(records, 1):
        print(f"Анализ {i}/{len(records)}, ID={vacancy_id(vacancy)}, ИИ={provider}", flush=True)
        findings = model.analyze(ai_context(vacancy))
        print(f"Проверенных замечаний от ИИ: {len(findings)}", flush=True)
        report.append(analyze_vacancy(vacancy, findings))
    return report


def process(args: argparse.Namespace, local: bool) -> None:
    if local:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
        vacancies = unique_records(records_from_json(payload))
        print(f"Локальный файл: {len(vacancies)} уникальных вакансий; портал не вызывается.")
    else:
        vacancies = fetch_vacancies(args.url, args.limit, args.page_size)
    report = build_report(vacancies, args.provider)
    write_atomic(Path(args.output), report)
    print(f"Отчёт сохранён: {args.output}, вакансий: {len(report)}")


def run_scheduled(args: argparse.Namespace) -> None:
    try:
        hour, minute = (int(v) for v in args.time.split(":"))
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            raise ValueError
    except ValueError as exc:
        raise ValueError("Время расписания должно иметь формат ЧЧ:ММ") from exc
    print(f"Расписание: ежедневно в {hour:02}:{minute:02} по времени компьютера. Остановка: Ctrl+C.", flush=True)
    while True:
        now = datetime.now()
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        print(f"Следующий запуск: {next_run:%d.%m.%Y %H:%M}", flush=True)
        while (seconds := (next_run - datetime.now()).total_seconds()) > 0:
            time.sleep(min(seconds, 30))
        try:
            process(args, local=False)
        except (AIError, ValueError, OSError, RuntimeError) as exc:
            print(f"Ошибка запуска по расписанию: {exc}. Последний отчёт сохранён.", file=sys.stderr, flush=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Прозрачная вакансия — команда Резонанс")
    sub = p.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--provider", choices=("gigachat", "ollama"), default=os.getenv("AI_PROVIDER", "gigachat"), help="Провайдер ИИ")
    shared.add_argument("--output", default=os.getenv("REPORT_PATH", "reports/report.json"), help="Путь к отчёту JSON")
    api = argparse.ArgumentParser(add_help=False)
    api.add_argument("--url", default=os.getenv("SOURCE_URL", DEFAULT_URL), help="Адрес API региона")
    api.add_argument("--limit", type=int, default=int(os.getenv("MAX_VACANCIES", "10")), help="Максимум вакансий")
    api.add_argument("--page-size", type=int, default=int(os.getenv("PAGE_SIZE", "5")), help="Размер страницы API, не более 100")
    sub.add_parser("run", parents=[shared, api], help="Получить вакансии через API и сформировать отчёт")
    local = sub.add_parser("check", parents=[shared], help="Проанализировать локальный JSON без обращения к порталу")
    local.add_argument("--input", required=True, help="Путь к JSON с ответом API")
    scheduled = sub.add_parser("schedule", parents=[shared, api], help="Запускать рабочий режим ежедневно")
    scheduled.add_argument("--time", default=os.getenv("SCHEDULE_TIME", "09:00"), help="Время запуска ЧЧ:ММ по времени компьютера")
    return p


def main() -> int:
    load_env()
    try:
        args = parser().parse_args()
        if args.command == "schedule":
            run_scheduled(args)
        else:
            process(args, local=args.command == "check")
        return 0
    except KeyboardInterrupt:
        print("\nРабота остановлена.")
        return 130
    except (AIError, ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Ошибка: {exc}. Предыдущий отчёт не изменён.", file=sys.stderr)
        return 1
