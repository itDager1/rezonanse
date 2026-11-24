import unittest
from unittest.mock import patch

from vacancy_analyzer.ai import AIAnalyzer, AIError, _extract_issues, validate_findings


class AIValidationTests(unittest.TestCase):
    def setUp(self):
        self.context = {
            "fields": {"график": [{"field": "schedule", "value": "гибкий график"}]},
            "description": [{"field": "description", "value": "Возможна работа по сменам."}],
        }
        self.good = {
            "topic": "график", "field": "schedule", "quote": "гибкий график",
            "problem": "Не указаны дни и часы работы.",
            "recommendation": "Уточните дни и часы работы.",
        }

    def test_keeps_only_source_backed_unique_findings(self):
        made_up = {**self.good, "quote": "полная занятость"}
        wrong_field = {**self.good, "field": "description"}
        wrong_topic = {**self.good, "topic": "юридический статус"}
        blank_quote = {**self.good, "quote": ""}
        actual_description = {**self.good, "field": "description", "quote": "по сменам"}
        found = validate_findings(self.context, [self.good, self.good, made_up, wrong_field,
                                                  wrong_topic, blank_quote, actual_description])
        self.assertEqual(found, [self.good, actual_description])

    def test_garbage_or_oversized_is_not_accepted(self):
        long_text = {**self.good, "problem": "a" * 1001}
        self.assertEqual(validate_findings(self.context, [None, {}, {**self.good, "problem": 42}, long_text]), [])

    def test_structured_field_must_match_topic_but_descriptions_are_shared(self):
        context = {
            "fields": {"график": [{"field": "schedule", "value": "гибкий график"}],
                       "оплата": [{"field": "salary_max", "value": "50000"}]},
            "description": [{"field": "duty", "value": "График по договорённости"}],
        }
        wrong = {**self.good, "field": "salary_max", "quote": "50000"}
        shared = {**self.good, "field": "duty", "quote": "по договорённости"}
        self.assertEqual(validate_findings(context, [wrong, shared]), [shared])

    def test_unexplained_generic_complaints_are_rejected(self):
        for phrase in ("Название графика неясно", "Некорректное описание графика работы"):
            with self.subTest(phrase=phrase):
                self.assertEqual(validate_findings(self.context, [{**self.good, "problem": phrase}]), [])
        self.assertEqual(validate_findings(self.context, [self.good]), [self.good])

    def test_response_json_parsing_and_failures(self):
        self.assertEqual(_extract_issues('```json\n{"issues": []}\n```'), [])
        for text in ('not json', '{"other": []}', '{"issues": {}}'):
            with self.subTest(text=text), self.assertRaises(AIError):
                _extract_issues(text)

    def test_schedule_details_in_other_field_remove_only_false_claims(self):
        context = {
            "fields": {"график": [{"field": "schedule", "value": "гибкий график"}]},
            "description": [{"field": "duty", "value": "Рабочие дни пн–пт; часы 09:00–18:00."}],
        }
        self.assertEqual(validate_findings(context, [self.good]), [])
        context["description"][0]["value"] = "Рабочие дни пн–пт."
        result = validate_findings(context, [self.good])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["problem"], "Не указаны конкретные часы работы.")
        self.assertIn("часы", result[0]["recommendation"])
        context["description"][0]["value"] = "Часы работы 09:00–18:00."
        result = validate_findings(context, [self.good])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["problem"], "Не указаны конкретные рабочие дни.")
        context["description"][0]["value"] = "Навыки работы с документами."
        self.assertEqual(validate_findings(context, [self.good]), [self.good])

    def test_does_not_infer_schedule_from_age_or_leave_a_combined_shift_finding(self):
        context = {
            "fields": {"график": [{"field": "schedule", "value": "гибкий график"}]},
            "description": [{"field": "requirement", "value": "Возраст с 18 до 60 лет."}],
        }
        self.assertEqual(validate_findings(context, [self.good]), [self.good])
        context["description"][0]["value"] = "Рабочие дни пн–пт, с 9 до 18."
        combined = {**self.good, "problem": "Не указаны дни и смены."}
        self.assertEqual(validate_findings(context, [combined]), [combined])

    def test_missing_credentials_cannot_silently_fall_back(self):
        with self.assertRaises(AIError):
            AIAnalyzer("gigachat", credentials="")

    def test_retry_is_bounded_and_succeeds_only_after_real_response(self):
        model = AIAnalyzer("ollama")
        with patch.object(model, "_analyze_once", side_effect=[AIError("bad JSON"), [self.good]]) as invoke, \
             patch("vacancy_analyzer.ai.time.sleep"):
            self.assertEqual(model.analyze(self.context), [self.good])
            self.assertEqual(invoke.call_count, 2)
        with patch.object(model, "_analyze_once", side_effect=AIError("bad JSON")) as invoke, \
             patch("vacancy_analyzer.ai.time.sleep"), self.assertRaisesRegex(AIError, "3"):
            model.analyze(self.context)
        self.assertEqual(invoke.call_count, 3)


if __name__ == "__main__":
    unittest.main()
