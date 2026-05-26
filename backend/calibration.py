from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Iterable

from interview_domain import Question
from interview_engine import evaluate_answer
from question_bank import build_dsa_questions


MIN_EXPECTED_CONCEPTS = 3
MIN_FOLLOW_UPS = 2
MIN_FAIL_CONDITIONS = 2
MIN_TIME_CAP_SECONDS = 240
MAX_TIME_CAP_SECONDS = 600
MIN_RUBRIC_DIMENSIONS = 5

SAMPLE_ANSWERS = {
    "strong": (
        "I would first clarify the input constraints and target output. Then I would choose a hash map, "
        "sliding window, queue, stack, or dynamic programming state depending on the structure of the problem. "
        "Because the maintained invariant keeps the scan correct, the approach is linear or near-linear time, "
        "with explicit space trade-offs. I would test empty input, duplicates, boundary cases, unreachable states, "
        "and explain why the algorithm remains correct."
    ),
    "partial": (
        "I would use a common data structure and scan through the input. I would try to keep track of what I have "
        "seen and mention the time complexity. I would also think about some edge cases."
    ),
    "weak": "I am not sure. I would maybe use loops and try examples.",
}


@dataclass
class QuestionCalibration:
    question_id: str
    title: str
    topic: str
    difficulty: str
    issue_count: int
    issues: list[str] = field(default_factory=list)
    sample_scores: dict[str, float | None] = field(default_factory=dict)
    sample_next_steps: dict[str, str] = field(default_factory=dict)
    score_spread: float | None = None


@dataclass
class CalibrationReport:
    question_count: int
    issue_count: int
    weak_question_ids: list[str]
    average_scores_by_sample: dict[str, float | None]
    questions: list[QuestionCalibration]

    def to_dict(self) -> dict:
        return asdict(self)


def average_dimension_score(scores) -> float | None:
    numeric_scores = [score.score for score in scores if score.score is not None]
    if not numeric_scores:
        return None
    return round(mean(numeric_scores), 2)


def inspect_question_metadata(question: Question) -> list[str]:
    issues: list[str] = []
    rubric_dimensions = question.rubric.dimensions

    if len(question.prompt.split()) < 18:
        issues.append("Prompt may be too shallow for a hiring-grade DSA interview.")
    if len(question.expected_concepts) < MIN_EXPECTED_CONCEPTS:
        issues.append("Expected concept coverage is thin.")
    if len(question.follow_ups) < MIN_FOLLOW_UPS:
        issues.append("Question has too few follow-ups for adaptive probing.")
    if len(rubric_dimensions) < MIN_RUBRIC_DIMENSIONS:
        issues.append("Rubric has fewer dimensions than the hiring scorecard expects.")
    if len(question.fail_conditions) < MIN_FAIL_CONDITIONS:
        issues.append("Question has too few fail conditions for reviewer calibration.")
    if question.time_cap_seconds < MIN_TIME_CAP_SECONDS:
        issues.append("Time cap may be too short for a structured verbal answer.")
    if question.time_cap_seconds > MAX_TIME_CAP_SECONDS:
        issues.append("Time cap may be too long for a focused interview question.")

    dimension_keys = {dimension.key for dimension in rubric_dimensions}
    required_dimensions = {
        "problem_understanding",
        "algorithm_choice",
        "correctness",
        "complexity",
        "edge_cases",
        "communication",
    }
    missing_dimensions = sorted(required_dimensions - dimension_keys)
    if missing_dimensions:
        issues.append(f"Rubric missing dimensions: {', '.join(missing_dimensions)}.")

    return issues


def score_question_samples(question: Question, sample_answers: dict[str, str]) -> tuple[dict[str, float | None], dict[str, str]]:
    sample_scores: dict[str, float | None] = {}
    sample_next_steps: dict[str, str] = {}

    for sample_name, answer_text in sample_answers.items():
        evaluation = evaluate_answer(question, answer_text, source_type="question")
        sample_scores[sample_name] = average_dimension_score(evaluation.dimension_scores)
        sample_next_steps[sample_name] = evaluation.next_step.value

    return sample_scores, sample_next_steps


def inspect_score_spread(sample_scores: dict[str, float | None]) -> tuple[float | None, list[str]]:
    issues: list[str] = []
    strong = sample_scores.get("strong")
    partial = sample_scores.get("partial")
    weak = sample_scores.get("weak")
    numeric_scores = [score for score in sample_scores.values() if score is not None]
    spread = round(max(numeric_scores) - min(numeric_scores), 2) if numeric_scores else None

    if strong is not None and weak is not None and strong - weak < 1.0:
        issues.append("Scoring spread between strong and weak sample answers is too narrow.")
    if strong is not None and strong < 3.4:
        issues.append("Strong sample answer scores below the expected hiring signal.")
    if weak is not None and weak > 3.0:
        issues.append("Weak sample answer scores above the expected weak-signal range.")
    if strong is not None and partial is not None and weak is not None and not (strong >= partial >= weak):
        issues.append("Sample answer ordering is inconsistent across strong, partial, and weak anchors.")

    return spread, issues


def calibrate_questions(
    questions: Iterable[Question],
    *,
    sample_answers: dict[str, str] | None = None,
) -> CalibrationReport:
    sample_answers = sample_answers or SAMPLE_ANSWERS
    question_reports: list[QuestionCalibration] = []

    for question in questions:
        issues = inspect_question_metadata(question)
        sample_scores, sample_next_steps = score_question_samples(question, sample_answers)
        score_spread, spread_issues = inspect_score_spread(sample_scores)
        issues.extend(spread_issues)

        question_reports.append(
            QuestionCalibration(
                question_id=question.id,
                title=question.title,
                topic=question.topic,
                difficulty=question.difficulty.value,
                issue_count=len(issues),
                issues=issues,
                sample_scores=sample_scores,
                sample_next_steps=sample_next_steps,
                score_spread=score_spread,
            )
        )

    issue_count = sum(report.issue_count for report in question_reports)
    averages: dict[str, float | None] = {}
    for sample_name in sample_answers:
        scores = [
            report.sample_scores[sample_name]
            for report in question_reports
            if report.sample_scores.get(sample_name) is not None
        ]
        averages[sample_name] = round(mean(scores), 2) if scores else None

    return CalibrationReport(
        question_count=len(question_reports),
        issue_count=issue_count,
        weak_question_ids=[report.question_id for report in question_reports if report.issue_count > 0],
        average_scores_by_sample=averages,
        questions=question_reports,
    )


def render_markdown_report(report: CalibrationReport) -> str:
    lines = [
        "# InterVo Calibration Report",
        "",
        f"- Questions inspected: {report.question_count}",
        f"- Total issues: {report.issue_count}",
        f"- Questions needing review: {len(report.weak_question_ids)}",
        "",
        "## Anchor Score Averages",
        "",
    ]

    for sample_name, average_score in report.average_scores_by_sample.items():
        score_text = "--" if average_score is None else f"{average_score:.2f}"
        lines.append(f"- {sample_name}: {score_text}")

    lines.extend(["", "## Question Findings", ""])

    for question in report.questions:
        lines.append(f"### {question.title}")
        lines.append("")
        lines.append(f"- ID: `{question.question_id}`")
        lines.append(f"- Topic: {question.topic}")
        lines.append(f"- Difficulty: {question.difficulty}")
        lines.append(f"- Score spread: {question.score_spread if question.score_spread is not None else '--'}")
        for sample_name, score in question.sample_scores.items():
            score_text = "--" if score is None else f"{score:.2f}"
            lines.append(f"- {sample_name} score: {score_text}; next step: {question.sample_next_steps[sample_name]}")
        if question.issues:
            lines.append("- Issues:")
            for issue in question.issues:
                lines.append(f"  - {issue}")
        else:
            lines.append("- Issues: none")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect question quality and scoring calibration.")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()

    report = calibrate_questions(build_dsa_questions())
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
        return
    print(render_markdown_report(report))


if __name__ == "__main__":
    main()
