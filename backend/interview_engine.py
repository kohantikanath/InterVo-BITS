from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable, List, Optional

from interview_domain import FollowUp, InterviewMode, Question


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

    def to_dict(self) -> dict:
        data = asdict(self)
        data["quality"] = self.quality.value
        data["clarity"] = self.clarity.value
        data["next_step"] = self.next_step.value
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

    return AnswerEvaluation(
        quality=quality,
        clarity=clarity,
        next_step=next_step,
        summary=summary,
        concept_hits=concept_hits,
        missing_concepts=missing_concepts,
        selected_follow_up_id=follow_up.id if follow_up else None,
        evidence=evidence,
    )


def build_review_acknowledgement(evaluation: AnswerEvaluation) -> str:
    if evaluation.next_step == NextStep.FOLLOW_UP:
        return "Answer evaluated. InterVo is ready to go one level deeper on this problem."
    if evaluation.next_step == NextStep.NEXT_QUESTION:
        return "Answer evaluated. InterVo is ready to move to the next question."
    return "Answer evaluated. This interview segment is ready to conclude."


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
