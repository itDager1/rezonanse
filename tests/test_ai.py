import ssl
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from vacancy_analyzer.ai import AIAnalyzer, AIError, _extract_issues, _read_json_response, validate_findings

# Public root certificate used only as valid test input; no key material is included.
TEST_CERT_PEM = """-----BEGIN CERTIFICATE-----
MIIDIzCCAqigAwIBAgIQFJgmZtx8zY9AU2d7uZnshTAKBggqhkjOPQQDAzCBlDEL
MAkGA1UEBhMCVVMxEzARBgNVBAgTCldhc2hpbmd0b24xEDAOBgNVBAcTB1JlZG1v
bmQxHjAcBgNVBAoTFU1pY3Jvc29mdCBDb3Jwb3JhdGlvbjE+MDwGA1UEAxM1TWlj
cm9zb2Z0IEVDQyBQcm9kdWN0IFJvb3QgQ2VydGlmaWNhdGUgQXV0aG9yaXR5IDIw
MTgwHhcNMTgwMjI3MjA0MjA4WhcNNDMwMjI3MjA1MDQ2WjCBlDELMAkGA1UEBhMC
VVMxEzARBgNVBAgTCldhc2hpbmd0b24xEDAOBgNVBAcTB1JlZG1vbmQxHjAcBgNV
BAoTFU1pY3Jvc29mdCBDb3Jwb3JhdGlvbjE+MDwGA1UEAxM1TWljcm9zb2Z0IEVD
QyBQcm9kdWN0IFJvb3QgQ2VydGlmaWNhdGUgQXV0aG9yaXR5IDIwMTgwdjAQBgcq
hkjOPQIBBgUrgQQAIgNiAATHERYqdh1Wjr65YmXUw8608MMw7I9t1245vMhJq6u4
40N41YEGXe/HfZ/O1rOQdd4MsJDeI7rI0T5n4BmpG4YxHl80Le4X/RX7fieKMqHq
yY/JfhjLLzssSHp9pvQBB6yjgbwwgbkwDgYDVR0PAQH/BAQDAgGGMA8GA1UdEwEB
/wQFMAMBAf8wHQYDVR0OBBYEFEPvcIe4nb/siBncxsRrdQ11NDMIMBAGCSsGAQQB
gjcVAQQDAgEAMGUGA1UdIAReMFwwBgYEVR0gADBSBgwrBgEEAYI3TIN9AQEwQjBA
BggrBgEFBQcCARY0aHR0cDovL3d3dy5taWNyb3NvZnQuY29tL3BraW9wcy9Eb2Nz
L1JlcG9zaXRvcnkuaHRtADAKBggqhkjOPQQDAwNpADBmAjEAocBJRF0yVSfMPpBu
JSKdJFubUTXHkUlJKqP5b08czd2c4bVXyZ7CIkWbBhVwHEW/AjEAxdMo63LHPrCs
Jwl/Yj1geeWS8UUquaUC5GC7/nornGCntZkU8rC+8LsFllZWj8Fo
-----END CERTIFICATE-----
"""


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

    def test_contradiction_requires_two_source_values(self):
        context = {
            "fields": {"требования": [{"field": "requirement.qualification", "value": "Ежедневное присутствие в офисе."}]},
            "description": [{"field": "duty", "value": "Работа полностью дистанционная."}],
        }
        finding = {"topic": "требования", "field": "requirement.qualification",
                   "quote": "Ежедневное присутствие в офисе.",
                   "problem": "Офисное присутствие противоречит дистанционной работе.",
                   "recommendation": "Уточните формат работы."}
        self.assertEqual(validate_findings(context, [finding]), [])

    def test_contradiction_implicit_comparison_requires_two_values(self):
        context = {"fields": {"график": [{"field": "schedule", "value": "2/2"}]}, "description": []}
        for problem in ("Указан график 2/2, но далее указан 5/2.",
                        "Указан график 2/2, хотя в другом поле указан 5/2.",
                        "Указан график 2/2, но в другом поле указан 5/2."):
            finding = {**self.good, "quote": "2/2", "problem": problem}
            with self.subTest(problem=problem):
                self.assertEqual(validate_findings(context, [finding]), [])

    def test_salary_zero_is_not_a_second_missing_salary_finding(self):
        context = {"fields": {"оплата": [{"field": "salary_min", "value": "0"},
                                        {"field": "salary_max", "value": "0"}]}, "description": []}
        finding = {"topic": "оплата", "field": "salary_min", "quote": "0",
                   "problem": "Отсутствует минимальная ставка.", "recommendation": "Укажите ставку."}
        self.assertEqual(validate_findings(context, [finding]), [])

    def test_missing_field_marker_is_not_used_as_a_quote(self):
        context = {"fields": {"требования": [{"field": "requirement.education", "value": "Не указано"}]}, "description": []}
        finding = {"topic": "требования", "field": "requirement.education", "quote": "Не указано",
                   "problem": "Отсутствует информация об образовании.", "recommendation": "Уточните образование."}
        self.assertEqual(validate_findings(context, [finding]), [])

    def test_does_not_invent_obligatory_lunch_break(self):
        context = {"fields": {"график": [{"field": "schedule", "value": "Понедельник–пятница, 10:00–19:00"}]}, "description": []}
        finding = {**self.good, "quote": "Понедельник–пятница, 10:00–19:00",
                   "problem": "Не указано время перерыва на обед.", "recommendation": "Укажите время перерыва."}
        self.assertEqual(validate_findings(context, [finding]), [])

    def test_shift_pattern_is_not_missing_weekly_shift_count(self):
        context = {"fields": {"график": [{"field": "schedule", "value": "Сменный график: 2/2, 08:00–20:00"}]}, "description": []}
        finding = {**self.good, "quote": "Сменный график: 2/2, 08:00–20:00",
                   "problem": "Не указано количество смен в неделе.", "recommendation": "Укажите точное количество смен в неделю."}
        self.assertEqual(validate_findings(context, [finding]), [])
        finding["problem"] = "Не указано количество смен в цикле и продолжительность каждой смены."
        self.assertEqual(validate_findings(context, [finding]), [])

    def test_specified_five_day_rhythm_is_not_missing_workdays(self):
        for schedule in ("Пятидневная рабочая неделя, 09:00–18:00",
                         "Полный рабочий день, 5/2, 08:30–17:30"):
            context = {"fields": {"график": [{"field": "schedule", "value": schedule}]}, "description": []}
            finding = {**self.good, "quote": schedule, "problem": "Не указаны конкретные рабочие дни.",
                       "recommendation": "Уточните рабочие дни."}
            with self.subTest(schedule=schedule):
                self.assertEqual(validate_findings(context, [finding]), [])

    def test_flexible_schedule_qualifies_five_day_week(self):
        context = {"fields": {"график": [{"field": "schedule", "value": "Гибкий график"}]},
                   "description": [{"field": "requirement.qualification", "value": "Пятидневная рабочая неделя с 09:00 до 18:00. Возможен гибкий график без уточнения правил."}]}
        finding = {**self.good, "quote": "Гибкий график", "problem": "Не указаны конкретные рабочие дни."}
        result = validate_findings(context, [finding])
        self.assertEqual(len(result), 1)
        self.assertIn("пятидневной", result[0]["problem"])
        self.assertIn("гибкий график", result[0]["problem"])

    def test_no_unrequested_flexible_schedule(self):
        schedule = "Полный рабочий день, понедельник–пятница, 09:00–18:00"
        context = {"fields": {"график": [{"field": "schedule", "value": schedule}]}, "description": []}
        finding = {**self.good, "quote": schedule,
                   "problem": "Неясно, предусмотрен ли гибкий график.",
                   "recommendation": "Уточните возможность гибкого графика."}
        self.assertEqual(validate_findings(context, [finding]), [])

    def test_flexible_schedule_question_remains_when_stated(self):
        schedule = "Гибкий график"
        context = {"fields": {"график": [{"field": "schedule", "value": schedule}]}, "description": []}
        finding = {**self.good, "quote": schedule,
                   "problem": "Не уточнено, какие часы можно выбирать при гибком графике.",
                   "recommendation": "Уточните часы гибкого графика."}
        self.assertEqual(validate_findings(context, [finding]), [finding])

    def test_schedule_is_not_office_schedule_just_because_hours_are_fixed(self):
        context = {"fields": {"график": [{"field": "schedule", "value": "Пятидневная рабочая неделя, 09:00–18:00"}]}, "description": []}
        finding = {**self.good, "quote": "Пятидневная рабочая неделя, 09:00–18:00",
                   "problem": "Указан офисный график, хотя работа дистанционная.",
                   "recommendation": "Уточните условия."}
        self.assertEqual(validate_findings(context, [finding]), [])
        finding["problem"] = "Указан только рабочий график в офисе, отсутствует информация о дистанционной работе."
        self.assertEqual(validate_findings(context, [finding]), [])

    def test_city_street_house_not_missing_region_for_a_readable_address(self):
        address = "г. Гусев, ул. Учебная, д. 4"
        context = {"fields": {"место работы": [{"field": "addresses.address[0].location", "value": address}]}, "description": []}
        finding = {"topic": "место работы", "field": "addresses.address[0].location", "quote": address,
                   "problem": "Отсутствует указание района или области, где находится город Гусев.",
                   "recommendation": "Добавьте информацию об области или районе."}
        self.assertEqual(validate_findings(context, [finding]), [])

        for problem in ("Нет информации о районе или удалённости от центра.", "Нет данных об области."):
            with self.subTest(problem=problem):
                self.assertEqual(validate_findings(context, [{**finding, "problem": problem}]), [])

    def test_full_address_does_not_require_floor_or_room(self):
        address = "г. Светлогорск, ул. Примерная, д. 3"
        context = {"fields": {"место работы": [{"field": "addresses.address[0].location", "value": address}]}, "description": []}
        finding = {"topic": "место работы", "field": "addresses.address[0].location", "quote": address,
                   "problem": "Отсутствует указание этажа или помещения.",
                   "recommendation": "Уточните этаж или номер помещения."}
        self.assertEqual(validate_findings(context, [finding]), [])
        for extra in ("этажа", "офиса", "кабинета"):
            with self.subTest(extra=extra):
                self.assertEqual(validate_findings(context, [{**finding, "problem": f"Не указан номер {extra}."}]), [])
        incomplete = "г. Светлогорск"
        context["fields"]["место работы"][0]["value"] = incomplete
        vague = {**finding, "quote": incomplete, "problem": "Не указан адрес рабочего места.",
                 "recommendation": "Уточните адрес рабочего места."}
        self.assertEqual(validate_findings(context, [vague]), [vague])

    def test_trial_period_must_be_mentioned_in_the_source(self):
        context = {"fields": {"тип занятости": [{"field": "employment", "value": "Полная занятость"}]}, "description": []}
        finding = {"topic": "тип занятости", "field": "employment", "quote": "Полная занятость",
                   "problem": "Нет информации о длительности испытательного срока.",
                   "recommendation": "Уточните продолжительность испытательного срока."}
        self.assertEqual(validate_findings(context, [finding]), [])
        context["description"].append({"field": "duty", "value": "Испытательный срок по договорённости."})
        self.assertEqual(validate_findings(context, [finding]), [finding])

    def test_missing_credentials_cannot_silently_fall_back(self):
        with self.assertRaises(AIError):
            AIAnalyzer("gigachat", credentials="")

    def test_ollama_ignores_missing_gigachat_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing.crts")
            model = AIAnalyzer("ollama", ca_bundle=missing)
        self.assertEqual(model.provider, "ollama")
        self.assertIsNone(model.ssl_context)

    def test_gigachat_missing_certificate_explains_path_and_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "сертификат.crts")
            with self.assertRaisesRegex(AIError, "не найден.*расширение"):
                AIAnalyzer("gigachat", credentials="dummy", ca_bundle=missing)

    def test_gigachat_invalid_certificate_reports_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            invalid = Path(tmp) / "certificate.crt"
            invalid.write_text("not a certificate", encoding="utf-8")
            with self.assertRaisesRegex(AIError, "PEM или DER"):
                AIAnalyzer("gigachat", credentials="dummy", ca_bundle=str(invalid))

    def test_gigachat_accepts_pem_and_der_on_unicode_paths_with_tls_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "Сертификаты"; folder.mkdir()
            pem = folder / "корень.crt"
            der = folder / "другой.crt"
            pem.write_bytes(b"\xef\xbb\xbf" + TEST_CERT_PEM.encode("ascii"))
            der.write_bytes(ssl.PEM_cert_to_DER_cert(TEST_CERT_PEM))
            for path in (pem, der):
                with self.subTest(path=path.name):
                    model = AIAnalyzer("gigachat", credentials="dummy", ca_bundle=str(path))
                    self.assertTrue(model.ssl_context.check_hostname)
                    self.assertEqual(model.ssl_context.verify_mode, ssl.CERT_REQUIRED)

    def test_gigachat_tls_verification_error_reports_certificate_guidance(self):
        request = urllib.request.Request("https://example.org/")
        failure = urllib.error.URLError(ssl.SSLCertVerificationError(1, "test verification failure"))
        with patch("vacancy_analyzer.ai.urllib.request.urlopen", side_effect=failure):
            with self.assertRaisesRegex(AIError, "проверить TLS-сертификат.*README"):
                _read_json_response(request, timeout=1, context=None, service="GigaChat")

    def test_gigachat_network_failure_does_not_claim_certificate_problem(self):
        request = urllib.request.Request("https://example.org/")
        failure = urllib.error.URLError(ConnectionRefusedError("test connection refused"))
        with patch("vacancy_analyzer.ai.urllib.request.urlopen", side_effect=failure):
            with self.assertRaises(AIError) as error:
                _read_json_response(request, timeout=1, context=None, service="GigaChat")
        self.assertIn("ошибка соединения", str(error.exception))
        self.assertNotIn("сертификат", str(error.exception))

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
