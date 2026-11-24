"""Тесты пагинации, локального режима и сохранности отчёта при сбоях."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vacancy_analyzer import cli


def record(identifier: str) -> dict:
    return {"vacancy": {"id": identifier, "job-name": "Тест"}}


class ProcessingTests(unittest.TestCase):
    def test_pagination_uses_page_indices_and_fixed_size(self):
        pages = [[record("1"), record("2")], [record("3"), record("4")],
                 [record("5"), record("6")]]
        calls = []

        def request(url, offset, limit):
            calls.append((offset, limit))
            return pages[offset]

        with patch.object(cli, "api_page", side_effect=request):
            result = cli.fetch_vacancies("https://example.test/api", 5, page_size=2)
        self.assertEqual(calls, [(0, 2), (1, 2), (2, 2)])
        self.assertEqual([v["id"] for v in result], ["1", "2", "3", "4", "5"])

    def test_check_processes_all_unique_local_records_without_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.json"
            output = Path(tmp) / "report.json"
            payload = {"status": "200", "results": {"vacancies": [
                record("1"), record("2"), record("1"), record("3")]}}
            source.write_text(json.dumps(payload), encoding="utf-8")
            args = argparse.Namespace(input=str(source), output=str(output), provider="ollama")

            def report(records, provider):
                return [{"id": r["id"], "findings": []} for r in records]

            with patch.object(cli, "api_page", side_effect=AssertionError("API вызван")), \
                 patch.object(cli, "build_report", side_effect=report):
                cli.process(args, local=True)
                first = output.read_bytes()
                cli.process(args, local=True)
            self.assertEqual(output.read_bytes(), first)
            rows = json.loads(first)
            self.assertEqual([r["id"] for r in rows], ["1", "2", "3"])

    def test_api_failure_leaves_last_successful_report_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            previous = b'[{"id":"previous","findings":[]}]\n'
            output.write_bytes(previous)
            args = argparse.Namespace(url="https://example.test/api", limit=3, page_size=2,
                                      output=str(output), provider="ollama")
            with patch.object(cli, "api_page", side_effect=[[record("1"), record("2")],
                                                             RuntimeError("connection failed")]), \
                 patch.object(cli, "build_report") as build:
                with self.assertRaises(RuntimeError):
                    cli.process(args, local=False)
            build.assert_not_called()
            self.assertEqual(output.read_bytes(), previous)

    def test_api_duplicate_ids_do_not_duplicate_output(self):
        pages = [[record("1"), record("2")], [record("2"), record("3")]]
        with patch.object(cli, "api_page", side_effect=pages):
            result = cli.fetch_vacancies("https://example.test/api", 3, page_size=2)
        self.assertEqual([r["id"] for r in result], ["1", "2", "3"])

    def test_error_status_is_not_misread_as_empty_vacancies(self):
        with self.assertRaises(ValueError):
            cli.records_from_json({"status": "500", "results": {"vacancies": []}})

    def test_schedule_calls_worker_at_configured_time(self):
        args = argparse.Namespace(time="09:00")
        current = [datetime(2026, 11, 24, 8, 59, 30),
                   datetime(2026, 11, 24, 8, 59, 30),
                   datetime(2026, 11, 24, 9, 0, 0)]
        with patch.object(cli, "datetime") as clock, \
             patch.object(cli.time, "sleep") as sleep, \
             patch.object(cli, "process", side_effect=KeyboardInterrupt) as process:
            clock.now.side_effect = current
            with self.assertRaises(KeyboardInterrupt):
                cli.run_scheduled(args)
        sleep.assert_called_once_with(30)
        process.assert_called_once_with(args, local=False)


if __name__ == "__main__":
    unittest.main()
