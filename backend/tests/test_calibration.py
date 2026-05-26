import unittest

from calibration import calibrate_questions, inspect_score_spread, render_markdown_report
from interview_domain import DifficultyLevel, FollowUp, Question, Rubric, RubricDimension
from question_bank import build_dsa_questions


class CalibrationTestCase(unittest.TestCase):
    def test_default_question_bank_calibrates_with_anchor_scores(self):
        report = calibrate_questions(build_dsa_questions())

        self.assertEqual(report.question_count, len(build_dsa_questions()))
        self.assertIn("strong", report.average_scores_by_sample)
        self.assertIn("weak", report.average_scores_by_sample)
        self.assertGreater(
            report.average_scores_by_sample["strong"],
            report.average_scores_by_sample["weak"],
        )
        self.assertEqual(len(report.questions), report.question_count)

    def test_metadata_inspection_flags_shallow_question(self):
        shallow_question = Question(
            id="too_shallow",
            topic="arrays",
            title="Too shallow",
            prompt="Solve two sum.",
            difficulty=DifficultyLevel.EASY,
            time_cap_seconds=120,
            expected_concepts=["hashing"],
            follow_ups=[],
            rubric=Rubric(
                dimensions=[
                    RubricDimension(
                        key="correctness",
                        label="Correctness",
                        description="Checks correctness only.",
                    )
                ]
            ),
            fail_conditions=[],
        )

        report = calibrate_questions([shallow_question])

        self.assertEqual(report.question_count, 1)
        self.assertEqual(report.weak_question_ids, ["too_shallow"])
        issues = report.questions[0].issues
        self.assertTrue(any("shallow" in issue for issue in issues))
        self.assertTrue(any("few follow-ups" in issue for issue in issues))
        self.assertTrue(any("Rubric missing dimensions" in issue for issue in issues))

    def test_score_spread_flags_inconsistent_anchor_scores(self):
        spread, issues = inspect_score_spread({"strong": 2.7, "partial": 3.2, "weak": 3.1})

        self.assertEqual(spread, 0.5)
        self.assertTrue(any("too narrow" in issue for issue in issues))
        self.assertTrue(any("ordering is inconsistent" in issue for issue in issues))

    def test_markdown_report_contains_question_findings(self):
        question = Question(
            id="calibration_render_question",
            topic="arrays",
            title="Render question",
            prompt=(
                "Given an array, explain how you would use a hash map to track values, handle duplicates, "
                "reason about correctness, and analyze complexity."
            ),
            difficulty=DifficultyLevel.MEDIUM,
            time_cap_seconds=300,
            expected_concepts=["hash map", "duplicates", "complexity"],
            follow_ups=[
                FollowUp(id="f1", prompt="What about memory?", intent="Trade-offs", trigger="standard"),
                FollowUp(id="f2", prompt="What about ordering?", intent="Output contract", trigger="standard"),
            ],
            rubric=Rubric(
                dimensions=[
                    RubricDimension(key="problem_understanding", label="Problem understanding", description=""),
                    RubricDimension(key="algorithm_choice", label="Algorithm choice", description=""),
                    RubricDimension(key="correctness", label="Correctness", description=""),
                    RubricDimension(key="complexity", label="Complexity", description=""),
                    RubricDimension(key="edge_cases", label="Edge cases", description=""),
                    RubricDimension(key="communication", label="Communication", description=""),
                ]
            ),
            fail_conditions=["No reasoning.", "No complexity."],
        )
        report = calibrate_questions([question])

        markdown = render_markdown_report(report)

        self.assertIn("# InterVo Calibration Report", markdown)
        self.assertIn("### Render question", markdown)
        self.assertIn("strong score", markdown)


if __name__ == "__main__":
    unittest.main()
