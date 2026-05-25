from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InterviewMode(str, Enum):
    HIRING = "hiring"
    PRACTICE = "practice"


class SessionStatus(str, Enum):
    DRAFT = "draft"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    REVIEW_PENDING = "review_pending"
    COMPLETED = "completed"
    ABORTED = "aborted"


class QuestionState(str, Enum):
    READY = "ready"
    ACTIVE = "active"
    WARNING = "warning"
    EXPIRED = "expired"
    SUBMITTED = "submitted"
    SCORED = "scored"
    ADVANCED = "advanced"


ALLOWED_SESSION_STATES = tuple(status.value for status in SessionStatus)
ALLOWED_QUESTION_STATES = tuple(state.value for state in QuestionState)


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class RecommendationBand(str, Enum):
    STRONG_HIRE = "strong_hire"
    HIRE = "hire"
    MIXED = "mixed"
    NO_HIRE = "no_hire"


class RubricDimension(BaseModel):
    key: str
    label: str
    description: str
    max_score: int = 5


class Rubric(BaseModel):
    dimensions: List[RubricDimension] = Field(default_factory=list)
    fail_conditions: List[str] = Field(default_factory=list)


class FollowUp(BaseModel):
    id: str
    prompt: str
    intent: str
    trigger: str


class Question(BaseModel):
    id: str
    topic: str
    title: str
    prompt: str
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    time_cap_seconds: int = 300
    expected_concepts: List[str] = Field(default_factory=list)
    follow_ups: List[FollowUp] = Field(default_factory=list)
    rubric: Rubric = Field(default_factory=Rubric)
    fail_conditions: List[str] = Field(default_factory=list)


class InterviewTemplate(BaseModel):
    id: str
    name: str
    mode: InterviewMode = InterviewMode.HIRING
    topic_family: str = "dsa"
    question_ids: List[str] = Field(default_factory=list)
    default_question_count: int = 6
    default_time_cap_seconds: int = 300


class ScoreEvidence(BaseModel):
    dimension: str
    summary: str
    transcript_excerpt: Optional[str] = None
    confidence: float = 0.0


class DimensionScore(BaseModel):
    dimension: str
    score: Optional[float] = None
    evidence: List[ScoreEvidence] = Field(default_factory=list)


class QuestionAttempt(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    question_id: Optional[str] = None
    question_title: Optional[str] = None
    state: QuestionState = QuestionState.READY
    source_type: str = "question"
    prompt_text: str = ""
    user_text: str = ""
    ai_text: str = ""
    selected_follow_up_id: Optional[str] = None
    evaluation_summary: str = ""
    recommended_next_step: str = ""
    expected_concepts_hit: List[str] = Field(default_factory=list)
    missing_concepts: List[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=utc_now)
    submitted_at: Optional[datetime] = None
    time_cap_seconds: Optional[int] = None
    scores: List[DimensionScore] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class FinalScorecard(BaseModel):
    recommendation: Optional[RecommendationBand] = None
    summary: str = ""
    dimension_scores: List[DimensionScore] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    unanswered_concerns: List[str] = Field(default_factory=list)
    evidence: List[ScoreEvidence] = Field(default_factory=list)
    generated_at: Optional[datetime] = None


class SessionEvent(BaseModel):
    at: datetime = Field(default_factory=utc_now)
    kind: str
    detail: str


class InterviewSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    template_id: Optional[str] = None
    mode: InterviewMode = InterviewMode.HIRING
    status: SessionStatus = SessionStatus.DRAFT
    current_question_id: Optional[str] = None
    current_attempt_id: Optional[str] = None
    last_user_text: str = ""
    last_ai_text: str = ""
    attempts: List[QuestionAttempt] = Field(default_factory=list)
    scorecard: FinalScorecard = Field(default_factory=FinalScorecard)
    events: List[SessionEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class InMemoryInterviewStore:
    """Temporary repository layer shaped for a future database-backed store."""

    def __init__(self):
        self.templates: Dict[str, InterviewTemplate] = {}
        self.questions: Dict[str, Question] = {}
        self.sessions: Dict[str, InterviewSession] = {}

    def create_session(
        self,
        *,
        template_id: Optional[str] = None,
        mode: InterviewMode = InterviewMode.HIRING,
        session_id: Optional[str] = None,
        status: SessionStatus = SessionStatus.DRAFT,
    ) -> InterviewSession:
        session = InterviewSession(
            id=session_id or str(uuid4()),
            template_id=template_id,
            mode=mode,
            status=status,
        )
        self.sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Optional[InterviewSession]:
        return self.sessions.get(session_id)

    def require_session(self, session_id: str) -> InterviewSession:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"Unknown interview session: {session_id}")
        return session

    def upsert_session(self, session: InterviewSession) -> InterviewSession:
        self.sessions[session.id] = session
        return session

    def get_or_create_session(
        self,
        session_id: str,
        *,
        template_id: Optional[str] = None,
        mode: InterviewMode = InterviewMode.HIRING,
        status: SessionStatus = SessionStatus.DRAFT,
    ) -> InterviewSession:
        existing = self.get_session(session_id)
        if existing is not None:
            return existing
        return self.create_session(
            template_id=template_id,
            mode=mode,
            session_id=session_id,
            status=status,
        )

    def record_event(self, session_id: str, kind: str, detail: str) -> InterviewSession:
        session = self.require_session(session_id)
        session.events.append(SessionEvent(kind=kind, detail=detail))
        return self.upsert_session(session)
