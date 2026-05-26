from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable, List, Optional

from interview_domain import DimensionScore, FollowUp, InterviewMode, Question, ScoreEvidence


class AnswerQuality(str, Enum):
    STRONG = "strong"
    PARTIAL = "partial"
    WEAK = "weak"


class AnswerClarity(str, Enum):
    CLEAR = "clear"
    MIXED = "mixed"
    UNCLEAR = "unclear"


class NextStep(str, Enum):
    FOLLOW_UP = "follow_up"
    NEXT_QUESTION = "next_question"
    COMPLETE = "complete"


@dataclass(frozen=True)
class AnswerEvaluation:
    quality: AnswerQuality
    clarity: AnswerClarity
    next_step: NextStep
    summary: str
    concept_hits: List[str] = field(default_factory=list)
    missing_concepts: List[str] = field(default_factory=list)
    selected_follow_up_id: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    dimension_scores: List[DimensionScore] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["quality"] = self.quality.value
        data["clarity"] = self.clarity.value
        data["next_step"] = self.next_step.value
        data["dimension_scores"] = [score.model_dump(mode="json") for score in self.dimension_scores]
        return data


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def tokenize(text: str) -> set[str]:
    cleaned = normalize_text(text)
    for char in ",.:;!?()[]{}-/":
        cleaned = cleaned.replace(char, " ")
    return {token for token in cleaned.split() if token}


def phrase_matches(text_tokens: set[str], phrase: str) -> bool:
    phrase_tokens = [token for token in tokenize(phrase) if token]
    if not phrase_tokens:
        return False
    overlap = sum(1 for token in phrase_tokens if token in text_tokens)
    required = max(1, (len(phrase_tokens) + 1) // 2)
    return overlap >= required


def extract_concept_hits(question: Question, user_text: str) -> tuple[List[str], List[str]]:
    tokens = tokenize(user_text)
    hits: List[str] = []
    misses: List[str] = []
    for concept in question.expected_concepts:
        if phrase_matches(tokens, concept):
            hits.append(concept)
        else:
            misses.append(concept)
    return hits, misses


def assess_clarity(user_text: str) -> AnswerClarity:
    normalized = normalize_text(user_text)
    words = normalized.split()
    filler_count = sum(word in {"um", "uh", "like", "basically", "maybe"} for word in words)
    if len(words) < 10 or filler_count >= 3:
        return AnswerClarity.UNCLEAR
    if len(words) < 20 or filler_count >= 1:
        return AnswerClarity.MIXED
    return AnswerClarity.CLEAR


def reasoning_evidence(user_text: str) -> List[str]:
    normalized = normalize_text(user_text)
    markers = {
        "because": "Uses causal reasoning.",
        "complexity": "Mentions complexity analysis.",
        "trade": "Mentions trade-offs.",
        "edge": "Mentions edge cases.",
        "invariant": "Mentions an invariant or maintained condition.",
        "visited": "Mentions tracked state such as visited nodes/cells.",
        "pointer": "Mentions pointer movement or traversal state.",
        "hash": "Mentions hash-based lookup or frequency tracking.",
        "queue": "Mentions queue-based traversal.",
        "stack": "Mentions stack-based state management.",
        "recurrence": "Mentions a recurrence or transition relation.",
    }
    evidence = [message for marker, message in markers.items() if marker in normalized]
    return evidence


def extract_excerpt(user_text: str, markers: Iterable[str]) -> str:
    normalized = normalize_text(user_text)
    for marker in markers:
        position = normalized.find(marker)
        if position != -1:
            end = min(len(user_text), position + 140)
            return user_text[position:end].strip()
    excerpt = " ".join(user_text.strip().split())
    return excerpt[:160].strip()


def clamp_score(value: float, minimum: float = 1.0, maximum: float = 5.0) -> float:
    return round(max(minimum, min(maximum, value)), 1)


def build_dimension_score(
    *,
    dimension: str,
    score: float,
    summary: str,
    transcript_excerpt: str,
    confidence: float,
) -> DimensionScore:
    return DimensionScore(
        dimension=dimension,
        score=clamp_score(score),
        evidence=[
            ScoreEvidence(
                dimension=dimension,
                summary=summary,
                transcript_excerpt=transcript_excerpt,
                confidence=round(confidence, 2),
            )
        ],
    )


def score_problem_understanding(question: Question, user_text: str, concept_hits: List[str], missing_concepts: List[str]) -> DimensionScore:
    hit_ratio = len(concept_hits) / max(1, len(question.expected_concepts))
    constraint_markers = ("given", "need", "target", "constraint", "input", "output")
    tokens = tokenize(user_text)
    constraint_mentions = sum(1 for marker in constraint_markers if marker in tokens)
    score = 2.0 + hit_ratio * 2.2 + min(0.8, constraint_mentions * 0.2)
    summary = (
        f"Recognized {len(concept_hits)} expected concepts; "
        f"missing {len(missing_concepts)} concept(s) and referenced {constraint_mentions} problem cues."
    )
    return build_dimension_score(
        dimension="problem_understanding",
        score=score,
        summary=summary,
        transcript_excerpt=extract_excerpt(user_text, constraint_markers),
        confidence=0.74 if concept_hits else 0.58,
    )


def score_algorithm_choice(user_text: str, evidence: List[str], quality: AnswerQuality) -> DimensionScore:
    algorithm_markers = ("hash", "pointer", "stack", "queue", "bfs", "dp", "heap", "bucket", "window")
    tokens = tokenize(user_text)
    strategy_mentions = sum(1 for marker in algorithm_markers if marker in tokens)
    quality_bonus = {
        AnswerQuality.STRONG: 1.2,
        AnswerQuality.PARTIAL: 0.6,
        AnswerQuality.WEAK: 0.1,
    }[quality]
    score = 1.8 + min(1.8, strategy_mentions * 0.45) + quality_bonus
    summary = f"Referenced {strategy_mentions} algorithmic strategy marker(s) and showed {quality.value} overall approach quality."
    return build_dimension_score(
        dimension="algorithm_choice",
        score=score,
        summary=summary,
        transcript_excerpt=extract_excerpt(user_text, algorithm_markers),
        confidence=0.78 if strategy_mentions else 0.55,
    )


def score_correctness(question: Question, user_text: str, concept_hits: List[str], evidence: List[str], clarity: AnswerClarity) -> DimensionScore:
    hit_ratio = len(concept_hits) / max(1, len(question.expected_concepts))
    reasoning_markers = ("because", "invariant", "visited", "maintain", "ensure", "correct")
    reasoning_mentions = sum(1 for marker in reasoning_markers if marker in normalize_text(user_text))
    clarity_bonus = {
        AnswerClarity.CLEAR: 0.8,
        AnswerClarity.MIXED: 0.4,
        AnswerClarity.UNCLEAR: 0.0,
    }[clarity]
    score = 1.6 + hit_ratio * 2.4 + min(0.8, reasoning_mentions * 0.25) + clarity_bonus
    summary = (
        f"Covered {len(concept_hits)} core correctness signal(s) with "
        f"{reasoning_mentions} explicit reasoning marker(s)."
    )
    return build_dimension_score(
        dimension="correctness",
        score=score,
        summary=summary,
        transcript_excerpt=extract_excerpt(user_text, reasoning_markers or evidence),
        confidence=0.8 if evidence else 0.6,
    )


def score_complexity(user_text: str) -> DimensionScore:
    complexity_markers = ("complexity", "o(", "linear", "constant", "space", "time", "trade")
    normalized = normalize_text(user_text)
    mentions = sum(1 for marker in complexity_markers if marker in normalized)
    score = 1.5 + min(2.7, mentions * 0.6)
    summary = f"Included {mentions} complexity or trade-off cue(s)."
    return build_dimension_score(
        dimension="complexity",
        score=score,
        summary=summary,
        transcript_excerpt=extract_excerpt(user_text, complexity_markers),
        confidence=0.82 if mentions >= 2 else 0.57,
    )


def score_edge_cases(user_text: str, missing_concepts: List[str]) -> DimensionScore:
    edge_markers = ("edge", "empty", "duplicate", "null", "boundary", "corner", "invalid", "unreachable")
    normalized = normalize_text(user_text)
    mentions = sum(1 for marker in edge_markers if marker in normalized)
    score = 1.5 + min(2.0, mentions * 0.7) + (0.3 if len(missing_concepts) <= 1 else 0.0)
    summary = f"Referenced {mentions} edge-case cue(s) with {len(missing_concepts)} remaining concept gap(s)."
    return build_dimension_score(
        dimension="edge_cases",
        score=score,
        summary=summary,
        transcript_excerpt=extract_excerpt(user_text, edge_markers),
        confidence=0.75 if mentions else 0.5,
    )


def score_communication(user_text: str, clarity: AnswerClarity) -> DimensionScore:
    words = normalize_text(user_text).split()
    sequencing_markers = ("first", "then", "next", "finally", "because")
    normalized = normalize_text(user_text)
    sequencing_mentions = sum(1 for marker in sequencing_markers if marker in normalized)
    clarity_base = {
        AnswerClarity.CLEAR: 4.2,
        AnswerClarity.MIXED: 3.1,
        AnswerClarity.UNCLEAR: 1.9,
    }[clarity]
    length_bonus = 0.4 if len(words) >= 35 else 0.0
    score = clarity_base + min(0.4, sequencing_mentions * 0.15) + length_bonus
    summary = f"Answer clarity was {clarity.value} with {sequencing_mentions} sequencing cue(s) across {len(words)} word(s)."
    return build_dimension_score(
        dimension="communication",
        score=score,
        summary=summary,
        transcript_excerpt=extract_excerpt(user_text, sequencing_markers),
        confidence=0.77 if clarity == AnswerClarity.CLEAR else 0.6,
    )


def build_dimension_scores(
    question: Question,
    user_text: str,
    *,
    concept_hits: List[str],
    missing_concepts: List[str],
    quality: AnswerQuality,
    clarity: AnswerClarity,
    evidence: List[str],
) -> List[DimensionScore]:
    raw_scores = {
        "problem_understanding": score_problem_understanding(question, user_text, concept_hits, missing_concepts),
        "algorithm_choice": score_algorithm_choice(user_text, evidence, quality),
        "correctness": score_correctness(question, user_text, concept_hits, evidence, clarity),
        "complexity": score_complexity(user_text),
        "edge_cases": score_edge_cases(user_text, missing_concepts),
        "communication": score_communication(user_text, clarity),
    }
    return [raw_scores[dimension.key] for dimension in question.rubric.dimensions if dimension.key in raw_scores]


def assess_quality(question: Question, concept_hits: List[str], clarity: AnswerClarity, evidence: List[str]) -> AnswerQuality:
    if not question.expected_concepts:
        return AnswerQuality.PARTIAL

    hit_ratio = len(concept_hits) / len(question.expected_concepts)
    if hit_ratio >= 0.6 and clarity != AnswerClarity.UNCLEAR and len(evidence) >= 2:
        return AnswerQuality.STRONG
    if hit_ratio >= 0.25 or len(evidence) >= 1:
        return AnswerQuality.PARTIAL
    return AnswerQuality.WEAK


def choose_follow_up(
    question: Question,
    *,
    quality: AnswerQuality,
    clarity: AnswerClarity,
    source_type: str,
) -> FollowUp | None:
    if source_type == "follow_up" or not question.follow_ups:
        return None

    def pick_matching(triggers: Iterable[str]) -> FollowUp | None:
        for follow_up in question.follow_ups:
            if any(trigger in follow_up.trigger for trigger in triggers):
                return follow_up
        return None

    if quality == AnswerQuality.STRONG:
        return pick_matching(("cleanly", "standard", "depth")) or question.follow_ups[0]
    if clarity == AnswerClarity.UNCLEAR:
        return pick_matching(("hand_wavy", "omits", "needs", "hardcoded")) or question.follow_ups[0]
    if quality == AnswerQuality.WEAK:
        return pick_matching(("comparison", "hand_wavy", "base_problem")) or question.follow_ups[0]
    return question.follow_ups[0]


def summarize_evaluation(
    *,
    question: Question,
    concept_hits: List[str],
    missing_concepts: List[str],
    quality: AnswerQuality,
    clarity: AnswerClarity,
) -> str:
    hit_text = ", ".join(concept_hits[:3]) if concept_hits else "few core concepts"
    missing_text = ", ".join(missing_concepts[:2]) if missing_concepts else "no major concept gaps"
    return (
        f"Quality={quality.value}; clarity={clarity.value}; "
        f"covered {hit_text}; remaining gaps: {missing_text}."
    )


def evaluate_answer(
    question: Question,
    user_text: str,
    *,
    source_type: str = "question",
) -> AnswerEvaluation:
    concept_hits, missing_concepts = extract_concept_hits(question, user_text)
    clarity = assess_clarity(user_text)
    evidence = reasoning_evidence(user_text)
    quality = assess_quality(question, concept_hits, clarity, evidence)
    follow_up = choose_follow_up(
        question,
        quality=quality,
        clarity=clarity,
        source_type=source_type,
    )

    next_step = NextStep.FOLLOW_UP if follow_up is not None else NextStep.NEXT_QUESTION
    summary = summarize_evaluation(
        question=question,
        concept_hits=concept_hits,
        missing_concepts=missing_concepts,
        quality=quality,
        clarity=clarity,
    )
    dimension_scores = build_dimension_scores(
        question,
        user_text,
        concept_hits=concept_hits,
        missing_concepts=missing_concepts,
        quality=quality,
        clarity=clarity,
        evidence=evidence,
    )

    return AnswerEvaluation(
        quality=quality,
        clarity=clarity,
        next_step=next_step,
        summary=summary,
        concept_hits=concept_hits,
        missing_concepts=missing_concepts,
        selected_follow_up_id=follow_up.id if follow_up else None,
        evidence=evidence,
        dimension_scores=dimension_scores,
    )


def build_review_acknowledgement(evaluation: AnswerEvaluation) -> str:
    if evaluation.next_step == NextStep.FOLLOW_UP:
        return "Answer evaluated. InterVo is ready to go one level deeper on this problem."
    if evaluation.next_step == NextStep.NEXT_QUESTION:
        return "Answer evaluated. InterVo is ready to move to the next question."
    return "Answer evaluated. This interview segment is ready to conclude."


def build_practice_feedback(question: Question, evaluation: AnswerEvaluation) -> dict:
    concept_hits = evaluation.concept_hits[:]
    missing_concepts = evaluation.missing_concepts[:]
    next_hint = (
        f"Revisit {missing_concepts[0]} and explain how it affects correctness or complexity."
        if missing_concepts
        else "Try tightening the proof of correctness and naming the key invariant."
    )
    rubric_signals = [
        {
            "dimension": score.dimension,
            "score": score.score,
            "summary": score.evidence[0].summary if score.evidence else "",
        }
        for score in evaluation.dimension_scores
    ]

    return {
        "mode": InterviewMode.PRACTICE.value,
        "question_id": question.id,
        "quality": evaluation.quality.value,
        "clarity": evaluation.clarity.value,
        "summary": evaluation.summary,
        "coaching": (
            "Practice review: compare your answer against the expected concepts, then revise the explanation "
            "to make the algorithm, correctness argument, complexity, and edge cases explicit."
        ),
        "concepts_hit": concept_hits,
        "missing_concepts": missing_concepts,
        "hint": next_hint,
        "solution_outline": [
            "State the core approach and the data structure or recurrence you would use.",
            "Name the invariant or transition that makes the approach correct.",
            "Give time and space complexity, including trade-offs.",
            "Call out edge cases before moving on.",
        ],
        "rubric_signals": rubric_signals,
    }


def build_interviewer_transition(
    *,
    next_step: NextStep,
    follow_up: FollowUp | None = None,
    next_question: Question | None = None,
    mode: InterviewMode = InterviewMode.HIRING,
) -> str:
    del mode

    if next_step == NextStep.FOLLOW_UP and follow_up is not None:
        return f"Let's stay with this problem and go one level deeper. {follow_up.prompt}"
    if next_step == NextStep.NEXT_QUESTION and next_question is not None:
        return f"Let's move to the next question. {next_question.prompt}"
    return "That concludes this interview segment."
