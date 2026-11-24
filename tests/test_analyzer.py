"""Проверки смысловых правил на небольших обезличенных примерах."""

import unittest

from vacancy_analyzer.analyzer import ai_context, analyze_vacancy, extract_vacancy, vacancy_id


def complete_vacancy():
    return {
        "id": "demo-123",
        "vac_url": "https://example.org/vacancy/demo-123",
        "job-name": "Инженер",
        "duty": "Монтаж и проверка оборудования",
        "requirement": {"education": "Среднее профессиональное", "experience": 0},
        "salary_min": 50000,
        "salary_max": 70000,
        "schedule": "Пятидневная рабочая неделя",
        "addresses": {"address": [{"location": "Калининград, ул. Лесная, 1"}]},
        "employment": "Полная занятость",
    }


class AnalyzerTest(unittest.TestCase):
    def test_complete_wrapped_api_record_has_no_findings(self):
        item = {"vacancy": complete_vacancy()}
        result = analyze_vacancy(item)
        self.assertEqual(result["id"], "demo-123")
        self.assertEqual(result["url"], "https://example.org/vacancy/demo-123")
        self.assertEqual(result["findings"], [])
        self.assertEqual(vacancy_id(item), "demo-123")
        self.assertEqual(extract_vacancy(item), item["vacancy"])
        context = ai_context(item)
        self.assertEqual(context["fields"]["место работы"][0]["field"], "addresses.address[0].location")

    def test_seven_missing_topics_with_empty_quotes(self):
        result = analyze_vacancy({"id": "empty"})
        self.assertEqual(len(result["findings"]), 7)
        self.assertEqual({f["topic"] for f in result["findings"]}, {
            "должность", "обязанности", "требования", "оплата", "график", "место работы", "тип занятости"
        })
        for finding in result["findings"]:
            self.assertEqual(finding["quote"], "")
            self.assertTrue(finding["field"] and finding["problem"] and finding["recommendation"])

    def test_conditions_in_description_and_structured_fields_are_combined(self):
        item = {
            "id": "description", "job-name": "Специалист", "duty": "Ведение документации",
            "description": "Требования: опыт работы с документами. Оплата 50 000 рублей. "
                           "График 5/2 с 09:00 до 18:00. Место работы: Калининград, ул. Мира, 10. "
                           "Полная занятость."
        }
        self.assertEqual(analyze_vacancy(item)["findings"], [])

    def test_free_text_in_duty_is_also_considered_for_other_topics(self):
        item = complete_vacancy()
        item.pop("schedule")
        item["duty"] += ". График работы 5/2."
        self.assertNotIn("график", [f["topic"] for f in analyze_vacancy(item)["findings"]])

    def test_flexible_schedule_needs_details_only_when_missing_everywhere(self):
        item = complete_vacancy()
        item["schedule"] = "Гибкий график"
        findings = analyze_vacancy(item)["findings"]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["topic"], "график")
        self.assertEqual(findings[0]["quote"], "Гибкий график")
        item["description"] = "График работы с 09:00 до 18:00."
        self.assertEqual(analyze_vacancy(item)["findings"], [])

    def test_weekly_hours_do_not_define_flexible_days_or_shifts(self):
        item = complete_vacancy()
        item["schedule"] = "Гибкий график, 40 часов в неделю"
        findings = analyze_vacancy(item)["findings"]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["topic"], "график")

    def test_inverted_salary_has_both_source_values(self):
        item = complete_vacancy()
        item["salary_min"], item["salary_max"] = 70000, 50000
        finding, = analyze_vacancy(item)["findings"]
        self.assertEqual(finding["topic"], "оплата")
        self.assertIn("salary_min", finding["field"])
        self.assertIn("salary_max", finding["field"])
        self.assertIn("70000", finding["quote"])
        self.assertIn("50000", finding["quote"])

    def test_zero_salary_without_description_is_missing(self):
        item = complete_vacancy()
        item["salary_min"] = item["salary_max"] = 0
        findings = analyze_vacancy(item)["findings"]
        self.assertEqual([(f["topic"], f["quote"]) for f in findings], [("оплата", "")])

    def test_vague_competitive_salary_requires_amount(self):
        for value in ("Достойная зарплата", "Конкурентоспособная заработная плата"):
            with self.subTest(value=value):
                item = complete_vacancy()
                item.pop("salary_min")
                item.pop("salary_max")
                item["salary"] = value
                findings = analyze_vacancy(item)["findings"]
                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0]["topic"], "оплата")
                self.assertEqual(findings[0]["quote"], value)

    def test_vague_adjective_does_not_hide_structured_or_description_amount(self):
        item = complete_vacancy()
        item["salary"] = "Достойная зарплата"
        self.assertEqual(analyze_vacancy(item)["findings"], [])
        item.pop("salary_min")
        item.pop("salary_max")
        item["description"] = "Оплата от 50 000 рублей."
        self.assertEqual(analyze_vacancy(item)["findings"], [])

    def test_unrelated_number_in_description_is_not_salary(self):
        item = complete_vacancy()
        item.pop("salary_min")
        item.pop("salary_max")
        item["salary"] = "Достойная зарплата"
        item["description"] = "Требования: опыт 3 года."
        findings = analyze_vacancy(item)["findings"]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["topic"], "оплата")

    def test_vague_salary_in_description_is_quoted_not_reported_missing(self):
        item = complete_vacancy()
        item.pop("salary_min")
        item.pop("salary_max")
        item["description"] = "Достойная зарплата. Требования: опыт 3 года."
        findings = analyze_vacancy(item)["findings"]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["field"], "description")
        self.assertIn("Достойная зарплата", findings[0]["quote"])

    def test_concrete_numeric_salary_field_counts_even_if_hourly(self):
        item = complete_vacancy()
        item.pop("salary_min")
        item.pop("salary_max")
        item["salary"] = 500
        item["description"] = "Достойная зарплата."
        self.assertEqual(analyze_vacancy(item)["findings"], [])

    def test_genuine_ai_quote_accepted_and_invented_quote_rejected(self):
        item = complete_vacancy()
        valid = {"topic": "обязанности", "field": "duty", "quote": "Монтаж",
                 "problem": "Не определено, какое оборудование имеется в виду.",
                 "recommendation": "Уточните виды оборудования."}
        invented = {**valid, "quote": "Работа с самолётами"}
        wrong_field = {**valid, "field": "job-name"}
        findings = analyze_vacancy(item, [valid, invented, wrong_field])["findings"]
        self.assertIn(valid, findings)
        self.assertNotIn(invented, findings)
        self.assertNotIn(wrong_field, findings)

    def test_model_cannot_claim_missing_field_that_has_value(self):
        fake = {"topic": "должность", "field": "job-name", "quote": "",
                "problem": "Название должности отсутствует.", "recommendation": "Укажите должность."}
        self.assertNotIn(fake, analyze_vacancy(complete_vacancy(), [fake])["findings"])

    def test_ai_finding_same_topic_field_and_quote_does_not_duplicate_rule(self):
        item = complete_vacancy()
        item["schedule"] = "Гибкий график"
        duplicate = {"topic": "график", "field": "schedule", "quote": "Гибкий график",
                     "problem": "Часы и дни не определены.", "recommendation": "Уточните время работы."}
        findings = analyze_vacancy(item, [duplicate])["findings"]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["field"], "schedule")

    def test_ai_findings_with_distinct_evidence_are_retained(self):
        item = complete_vacancy()
        item["schedule"] = "Гибкий график"
        item["duty"] = "Разные поручения"
        ai_finding = {"topic": "обязанности", "field": "duty", "quote": "Разные поручения",
                      "problem": "Обязанности не конкретизированы.", "recommendation": "Уточните задачи."}
        findings = analyze_vacancy(item, [ai_finding])["findings"]
        self.assertEqual(len(findings), 2)
        self.assertIn(ai_finding, findings)

    def test_experience_value_zero_and_five_year_requirement_both_reported(self):
        item = complete_vacancy()
        item["requirement"]["qualification"] = "Требуется не менее 5 лет опыта работы аналитиком данных."
        result = analyze_vacancy(item)["findings"]
        issue, = [f for f in result if f["topic"] == "требования"]
        self.assertIn("requirement.experience", issue["field"])
        self.assertIn("requirement.qualification", issue["field"])
        self.assertIn("0", issue["quote"])
        self.assertIn(item["requirement"]["qualification"], issue["quote"])

    def test_experience_zero_and_optional_experience_is_not_contradiction(self):
        item = complete_vacancy()
        item["requirement"]["qualification"] = "Опыт не обязателен."
        self.assertEqual(analyze_vacancy(item)["findings"], [])
        item["requirement"]["qualification"] = "Не менее 5 лет опыта не требуется."
        self.assertEqual(analyze_vacancy(item)["findings"], [])

    def test_explicit_remote_vs_daily_office_has_both_sources(self):
        item = complete_vacancy()
        item["duty"] += ". Работа полностью дистанционная из любого города."
        item["requirement"]["qualification"] = "Обязательное ежедневное присутствие в офисе в Калининграде к 09:00."
        issue, = [f for f in analyze_vacancy(item)["findings"] if f["topic"] == "место работы"]
        self.assertIn("duty", issue["field"])
        self.assertIn("requirement.qualification", issue["field"])
        self.assertIn(item["duty"], issue["quote"])
        self.assertIn(item["requirement"]["qualification"], issue["quote"])

    def test_remote_without_required_office_does_not_create_contradiction(self):
        item = complete_vacancy()
        item["duty"] += ". Работа полностью дистанционная из любого города."
        item["requirement"]["qualification"] = "Не требуется обязательное присутствие в офисе."
        self.assertNotIn("место работы", [f["topic"] for f in analyze_vacancy(item)["findings"]])

    def test_hybrid_work_is_not_claimed_to_be_fully_remote(self):
        item = complete_vacancy()
        item["duty"] += ". Дистанционная работа возможна по согласованию."
        item["requirement"]["qualification"] = "Обязательное ежедневное присутствие в офисе."
        self.assertNotIn("место работы", [f["topic"] for f in analyze_vacancy(item)["findings"]])
        item["duty"] = "Полностью дистанционная работа или обязательное присутствие в офисе по выбору сотрудника."
        self.assertNotIn("место работы", [f["topic"] for f in analyze_vacancy(item)["findings"]])

    def test_shift_schedule_without_detail_has_actionable_finding(self):
        item = complete_vacancy()
        item["schedule"] = "Сменный график"
        issue, = analyze_vacancy(item)["findings"]
        self.assertEqual(issue["topic"], "график")
        self.assertEqual(issue["quote"], "Сменный график")
        item["requirement"]["qualification"] = "Смены 2/2 с 08:00 до 20:00."
        self.assertEqual(analyze_vacancy(item)["findings"], [])

    def test_unspecified_other_duties_quote_original(self):
        item = complete_vacancy()
        item["duty"] = "Работа с документами и другие задачи по необходимости."
        issue, = analyze_vacancy(item)["findings"]
        self.assertEqual(issue["topic"], "обязанности")
        self.assertEqual(issue["field"], "duty")
        self.assertEqual(issue["quote"], "другие задачи по необходимости")

    def test_vague_director_tasks_and_generic_specialist_name(self):
        item = complete_vacancy()
        item["duty"] = "Выполнение различных поручений руководителя."
        issue, = analyze_vacancy(item)["findings"]
        self.assertEqual((issue["topic"], issue["quote"]), ("обязанности", item["duty"]))
        item["job-name"] = "Специалист"
        topics = [f["topic"] for f in analyze_vacancy(item)["findings"]]
        self.assertIn("должность", topics)
        self.assertIn("обязанности", topics)

    def test_duty_details_elsewhere_do_not_require_repeat(self):
        item = complete_vacancy()
        item["job-name"] = "Специалист"
        item["duty"] = "Выполнение различных поручений руководителя."
        item["description"] = "Должность: специалист по документообороту. Обязанности: ведение архива и обработка входящих писем."
        self.assertEqual(analyze_vacancy(item)["findings"], [])

    def test_specific_job_elsewhere_prevents_generic_title_finding(self):
        item = complete_vacancy()
        item["job-name"] = "Специалист"
        item["duty"] = "Работа с документами и другие задачи по необходимости."
        item["description"] = "Должность: специалист по кадрам."
        topics = [f["topic"] for f in analyze_vacancy(item)["findings"]]
        self.assertNotIn("должность", topics)
        self.assertIn("обязанности", topics)

    def test_workplace_deferred_with_only_city_needs_clarification(self):
        item = complete_vacancy()
        item["addresses"]["address"][0]["location"] = "г. Калининград"
        item["requirement"]["qualification"] = "Формат и адрес рабочего места уточняются при собеседовании."
        issue, = [f for f in analyze_vacancy(item)["findings"] if f["topic"] == "место работы"]
        self.assertEqual(issue["field"], "requirement.qualification")
        self.assertEqual(issue["quote"], item["requirement"]["qualification"])

    def test_workplace_deferred_does_not_override_real_address_or_remote_format(self):
        item = complete_vacancy()
        item["requirement"]["qualification"] = "Формат и адрес рабочего места уточняются при собеседовании."
        self.assertNotIn("место работы", [f["topic"] for f in analyze_vacancy(item)["findings"]])
        item["addresses"]["address"][0]["location"] = "г. Калининград"
        item["description"] = "Работа полностью дистанционная."
        self.assertNotIn("место работы", [f["topic"] for f in analyze_vacancy(item)["findings"]])

    def test_conflicting_shift_schemes_show_both_source_values(self):
        item = complete_vacancy()
        item["schedule"] = "2/2"
        item["description"] = "График работы 5/2."
        issue, = [f for f in analyze_vacancy(item)["findings"] if f["topic"] == "график"]
        self.assertEqual(issue["field"], "schedule, description")
        self.assertEqual(issue["quote"], "2/2; График работы 5/2.")

    def test_repeated_shift_scheme_and_unrelated_fraction_are_not_contradictions(self):
        item = complete_vacancy()
        item["schedule"] = "2/2"
        for value in ("График работы 2/2.", "Отчёт за 5/2 предоставляется руководителю.",
                      "График работы 5/2 или 2/2 по выбору."):
            with self.subTest(value=value):
                item["description"] = value
                self.assertNotIn("график", [f["topic"] for f in analyze_vacancy(item)["findings"]])


if __name__ == "__main__":
    unittest.main()
