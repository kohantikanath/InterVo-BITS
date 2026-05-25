import os
import shutil
from datetime import timedelta

import static_ffmpeg

static_ffmpeg.add_paths()

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from gtts import gTTS
from interview_domain import (
    ALLOWED_QUESTION_STATES,
    ALLOWED_SESSION_STATES,
    DimensionScore,
    FinalScorecard,
    InMemoryInterviewStore,
    InterviewMode,
    InterviewSession,
    QuestionAttempt,
    QuestionScorecardEntry,
    QuestionState,
    Question,
    RecommendationBand,
    ScoreEvidence,
    SessionStatus,
    utc_now,
)
from interview_engine import (
    NextStep,
    build_interviewer_transition,
    build_review_acknowledgement,
    evaluate_answer,
)
from interview_policy import build_system_prompt, evaluate_candidate_message, get_interview_policy
from litellm import completion
from pydantic import BaseModel
from pydub import AudioSegment
from question_bank import default_template_id_for_mode, seed_question_bank

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash")
DEFAULT_SESSION_TIME_CAP_SECONDS = 300
DEFAULT_WARNING_THRESHOLD_SECONDS = 30
# Set ffmpeg paths required by pydub.
AudioSegment.converter = shutil.which("ffmpeg")
AudioSegment.ffmpeg = shutil.which("ffmpeg")
AudioSegment.ffprobe = shutil.which("ffprobe")

app = FastAPI()

whisper_model = None
interview_store = InMemoryInterviewStore()
seed_question_bank(interview_store)
LEGACY_SESSION_ID = "legacy-singleton-session"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_legacy_session():
    return interview_store.get_or_create_session(
        LEGACY_SESSION_ID,
        template_id=default_template_id_for_mode(InterviewMode.HIRING),
        mode=InterviewMode.HIRING,
        status=SessionStatus.READY,
    )


def ensure_legacy_attempt() -> QuestionAttempt:
    session = get_legacy_session()
    if session.attempts and session.current_attempt_id == session.attempts[-1].id:
        return session.attempts[-1]

    attempt = QuestionAttempt(
        question_id="legacy-live-question",
        question_title="Legacy interview loop",
        state=QuestionState.READY,
    )
    session.current_attempt_id = attempt.id
    session.attempts.append(attempt)
    interview_store.upsert_session(session)
    return attempt


def update_legacy_transcript(*, user_text: str | None = None, ai_text: str | None = None) -> None:
    session = get_legacy_session()
    attempt = ensure_legacy_attempt()

    if user_text is not None:
        session.last_user_text = user_text
        attempt.user_text = user_text
        if attempt.state == QuestionState.READY and user_text:
            attempt.state = QuestionState.SUBMITTED
        if not user_text and attempt.state == QuestionState.READY:
            attempt.state = QuestionState.READY
    if ai_text is not None:
        session.last_ai_text = ai_text
        attempt.ai_text = ai_text
        if ai_text and attempt.state == QuestionState.SUBMITTED:
            attempt.state = QuestionState.SCORED

    interview_store.record_event(
        session.id,
        "legacy_transcript_updated",
        f"user_text={'yes' if user_text is not None else 'no'}, ai_text={'yes' if ai_text is not None else 'no'}",
    )
    interview_store.upsert_session(session)


def require_session_or_404(session_id: str) -> InterviewSession:
    try:
        return interview_store.require_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def create_session_attempt(
    session: InterviewSession,
    *,
    question_id: str,
    question_title: str,
    prompt: str,
    time_cap_seconds: int = DEFAULT_SESSION_TIME_CAP_SECONDS,
    source_type: str = "question",
    selected_follow_up_id: str | None = None,
) -> QuestionAttempt:
    started_at = utc_now()
    warning_threshold_seconds = min(DEFAULT_WARNING_THRESHOLD_SECONDS, max(10, time_cap_seconds // 4))
    attempt = QuestionAttempt(
        question_id=question_id,
        question_title=question_title,
        source_type=source_type,
        prompt_text=prompt,
        ai_text=prompt,
        selected_follow_up_id=selected_follow_up_id,
        state=QuestionState.ACTIVE,
        started_at=started_at,
        time_cap_seconds=time_cap_seconds,
        warning_threshold_seconds=warning_threshold_seconds,
        expires_at=started_at + timedelta(seconds=time_cap_seconds),
    )
    session.current_question_id = question_id
    session.current_attempt_id = attempt.id
    session.last_ai_text = prompt
    session.attempts.append(attempt)
    return attempt


def get_current_attempt(session: InterviewSession) -> QuestionAttempt | None:
    if not session.current_attempt_id:
        return None

    for attempt in reversed(session.attempts):
        if attempt.id == session.current_attempt_id:
            return attempt

    return None


def build_session_payload(session: InterviewSession) -> dict:
    current_attempt = get_current_attempt(session)
    return {
        "session": session.model_dump(mode="json"),
        "current_attempt": current_attempt.model_dump(mode="json") if current_attempt else None,
        "timer": build_attempt_timer_snapshot(current_attempt) if current_attempt else None,
        "allowed_session_states": ALLOWED_SESSION_STATES,
        "allowed_question_states": ALLOWED_QUESTION_STATES,
    }


def build_opening_question_prompt() -> str:
    return (
        "Start a DSA hiring interview. Introduce yourself as InterVo, keep a professional tone, "
        "and ask one opening data structures or algorithms question in 2-3 sentences."
    )


def build_next_question_prompt(session: InterviewSession) -> str:
    answered_count = sum(1 for attempt in session.attempts if attempt.user_text.strip())
    return (
        f"The candidate has answered {answered_count} question(s). "
        f"The latest answer was: {session.last_user_text or 'No answer captured.'} "
        "Ask the next DSA interview question in 2-3 sentences without revealing the solution."
    )


def build_legacy_answer_prompt(user_text: str) -> str:
    return (
        f"The candidate answered: {user_text}. "
        "Respond as a strict DSA interviewer: briefly acknowledge the attempt, ask one targeted follow-up "
        "or next question, and do not reveal the answer, teach the solution, or discuss grading. "
        "Keep the response to 2-3 sentences."
    )


def collect_prior_user_texts(session: InterviewSession) -> list[str]:
    return [attempt.user_text for attempt in session.attempts if attempt.user_text.strip()]


def get_template_for_session(session: InterviewSession):
    template_id = session.template_id or default_template_id_for_mode(session.mode)
    template = interview_store.templates.get(template_id)
    if template is None:
        raise HTTPException(status_code=404, detail=f"Unknown interview template: {template_id}")
    return template


def get_question_for_session_index(session: InterviewSession, question_index: int):
    template = get_template_for_session(session)
    max_questions = min(template.default_question_count, len(template.question_ids))
    if question_index < 0 or question_index >= max_questions:
        return None

    question_id = template.question_ids[question_index]
    question = interview_store.questions.get(question_id)
    if question is None:
        raise HTTPException(status_code=404, detail=f"Unknown interview question: {question_id}")
    return question


def get_question_by_attempt(attempt: QuestionAttempt) -> Question:
    question = interview_store.questions.get(attempt.question_id or "")
    if question is None:
        raise HTTPException(status_code=404, detail=f"Unknown interview question: {attempt.question_id}")
    return question


def get_follow_up_by_id(question: Question, follow_up_id: str | None):
    if not follow_up_id:
        return None
    for follow_up in question.follow_ups:
        if follow_up.id == follow_up_id:
            return follow_up
    raise HTTPException(status_code=404, detail=f"Unknown follow-up question: {follow_up_id}")


def count_primary_attempts(session: InterviewSession) -> int:
    return sum(1 for attempt in session.attempts if attempt.source_type == "question")


def latest_attempts_by_question(session: InterviewSession) -> list[QuestionAttempt]:
    latest: dict[str, QuestionAttempt] = {}
    for attempt in session.attempts:
        if attempt.question_id:
            latest[attempt.question_id] = attempt
    return list(latest.values())


def latest_attempt_by_key(session: InterviewSession) -> dict[tuple[str, str], QuestionAttempt]:
    latest: dict[tuple[str, str], QuestionAttempt] = {}
    for attempt in session.attempts:
        if attempt.question_id:
            latest[(attempt.question_id, attempt.source_type)] = attempt
    return latest


def is_attempt_rubric_complete(question: Question, attempt: QuestionAttempt) -> bool:
    dimension_keys = [dimension.key for dimension in question.rubric.dimensions]
    score_map = {score.dimension: score for score in attempt.scores}
    for key in dimension_keys:
        dimension_score = score_map.get(key)
        if dimension_score is None or dimension_score.score is None or not dimension_score.evidence:
            return False
        if not any(evidence.summary and evidence.transcript_excerpt for evidence in dimension_score.evidence):
            return False
    return True


def aggregate_dimension_scores(attempts: list[QuestionAttempt]) -> list[DimensionScore]:
    aggregated: dict[str, list[DimensionScore]] = {}
    for attempt in attempts:
        for score in attempt.scores:
            aggregated.setdefault(score.dimension, []).append(score)

    merged_scores: list[DimensionScore] = []
    for dimension, dimension_scores in aggregated.items():
        numeric_scores = [score.score for score in dimension_scores if score.score is not None]
        average_score = round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else None
        evidence: list[ScoreEvidence] = []
        for score in dimension_scores:
            evidence.extend(score.evidence[:1])
        merged_scores.append(
            DimensionScore(
                dimension=dimension,
                score=average_score,
                evidence=evidence[:3],
            )
        )
    return sorted(merged_scores, key=lambda score: score.dimension)


def average_dimension_score(scores: list[DimensionScore]) -> float | None:
    numeric_scores = [score.score for score in scores if score.score is not None]
    if not numeric_scores:
        return None
    return round(sum(numeric_scores) / len(numeric_scores), 2)


def build_recommendation(
    average_score: float,
    *,
    low_dimension_count: int = 0,
    critical_dimension_scores: dict[str, float] | None = None,
) -> RecommendationBand:
    critical_dimension_scores = critical_dimension_scores or {}
    lowest_critical = min(critical_dimension_scores.values(), default=average_score)

    if average_score >= 4.2 and lowest_critical >= 3.8 and low_dimension_count == 0:
        return RecommendationBand.STRONG_HIRE
    if average_score >= 3.4 and lowest_critical >= 3.0 and low_dimension_count <= 1:
        return RecommendationBand.HIRE
    if average_score >= 2.6:
        return RecommendationBand.MIXED
    return RecommendationBand.NO_HIRE


def humanize_dimension(dimension_key: str) -> str:
    return dimension_key.replace("_", " ")


def question_average(attempt: QuestionAttempt) -> float | None:
    return average_dimension_score(attempt.scores)


def build_question_scorecard_entry(attempt: QuestionAttempt) -> QuestionScorecardEntry:
    evidence: list[ScoreEvidence] = []
    for score in attempt.scores:
        evidence.extend(score.evidence[:1])

    return QuestionScorecardEntry(
        question_id=attempt.question_id or "",
        question_title=attempt.question_title or "",
        source_type=attempt.source_type,
        state=attempt.state,
        prompt_text=attempt.prompt_text,
        answer_text=attempt.user_text,
        evaluation_summary=attempt.evaluation_summary,
        recommended_next_step=attempt.recommended_next_step,
        score_average=question_average(attempt),
        dimension_scores=attempt.scores,
        evidence=evidence[:3],
        concepts_demonstrated=attempt.expected_concepts_hit,
        missing_concepts=attempt.missing_concepts,
        started_at=attempt.started_at,
        submitted_at=attempt.submitted_at,
        expired=attempt.state == QuestionState.EXPIRED,
        lock_reason=attempt.lock_reason,
    )


def build_question_summaries(session: InterviewSession, template_question_ids: list[str]) -> list[QuestionScorecardEntry]:
    latest_attempts = latest_attempt_by_key(session)
    ordered_entries: list[QuestionScorecardEntry] = []

    for question_id in template_question_ids:
        primary_attempt = latest_attempts.get((question_id, "question"))
        if primary_attempt is not None:
            ordered_entries.append(build_question_scorecard_entry(primary_attempt))

        follow_up_attempt = latest_attempts.get((question_id, "follow_up"))
        if follow_up_attempt is not None:
            ordered_entries.append(build_question_scorecard_entry(follow_up_attempt))

    return ordered_entries


def build_unanswered_concerns(
    attempts: list[QuestionAttempt],
    question_entries: list[QuestionScorecardEntry],
) -> list[str]:
    concerns: list[str] = []
    for attempt in attempts:
        if attempt.state == QuestionState.EXPIRED:
            concerns.append(
                f"{attempt.question_title or attempt.question_id}: time expired before a complete evaluation."
            )
            continue

        if attempt.missing_concepts:
            focus = ", ".join(attempt.missing_concepts[:2])
            concerns.append(
                f"{attempt.question_title or attempt.question_id}: candidate did not clearly cover {focus}."
            )

    if not concerns:
        low_scoring_entries = [
            entry
            for entry in question_entries
            if entry.score_average is not None and entry.score_average < 3.0
        ]
        for entry in low_scoring_entries[:2]:
            concerns.append(
                f"{entry.question_title}: overall signal remained below the hiring bar and needs recruiter review."
            )

    return concerns[:4]


def build_scorecard_summary(
    session: InterviewSession,
    scorecard: FinalScorecard,
    strongest_dimensions: list[str],
    weakest_dimensions: list[str],
) -> str:
    if scorecard.recommendation_ready and scorecard.overall_average is not None:
        strength_text = ", ".join(strongest_dimensions[:2]) if strongest_dimensions else "balanced performance"
        risk_text = ", ".join(weakest_dimensions[:2]) if weakest_dimensions else "no major weak spots"
        return (
            f"Completed {scorecard.attempts_graded}/{scorecard.questions_expected} scored questions with an overall "
            f"average of {scorecard.overall_average:.2f}. Strongest areas: {strength_text}. "
            f"Primary risks: {risk_text}."
        )

    if session.status != SessionStatus.COMPLETED:
        return (
            f"Interview still in progress. {scorecard.attempts_graded}/{scorecard.questions_expected} primary "
            "question rubric(s) are fully scored so far."
        )

    return (
        f"Interview finished, but the recruiter scorecard is incomplete. "
        f"{scorecard.attempts_graded}/{scorecard.questions_expected} primary question rubric(s) have complete scoring."
    )


def refresh_scorecard(session: InterviewSession) -> FinalScorecard:
    template = get_template_for_session(session)
    expected_question_ids = template.question_ids[: min(template.default_question_count, len(template.question_ids))]
    latest_attempts = latest_attempts_by_question(session)
    expected_questions = len(expected_question_ids)
    complete_attempts: list[QuestionAttempt] = []
    rubric_complete = True

    for attempt in latest_attempts:
        question = get_question_by_attempt(attempt)
        if attempt.state == QuestionState.EXPIRED:
            rubric_complete = False
            continue
        if not is_attempt_rubric_complete(question, attempt):
            rubric_complete = False
            continue
        complete_attempts.append(attempt)

    scorecard = FinalScorecard(
        grading_complete=rubric_complete and len(complete_attempts) == expected_questions,
        recommendation_ready=(
            session.status == SessionStatus.COMPLETED
            and rubric_complete
            and len(complete_attempts) == expected_questions
        ),
        attempts_graded=len(complete_attempts),
        questions_expected=expected_questions,
    )
    scorecard.question_summaries = build_question_summaries(session, expected_question_ids)

    if complete_attempts:
        scorecard.dimension_scores = aggregate_dimension_scores(complete_attempts)
        scorecard.evidence = [evidence for score in scorecard.dimension_scores for evidence in score.evidence[:1]][:8]
        scorecard.strengths = [
            humanize_dimension(score.dimension)
            for score in scorecard.dimension_scores
            if score.score is not None and score.score >= 4.0
        ]
        scorecard.risks = [
            humanize_dimension(score.dimension)
            for score in scorecard.dimension_scores
            if score.score is not None and score.score <= 3.0
        ]
        scorecard.overall_average = average_dimension_score(scorecard.dimension_scores)
        scorecard.unanswered_concerns = build_unanswered_concerns(
            latest_attempts,
            scorecard.question_summaries,
        )

    strongest_dimensions = scorecard.strengths
    weakest_dimensions = scorecard.risks

    if scorecard.recommendation_ready and scorecard.dimension_scores and scorecard.overall_average is not None:
        critical_dimensions = {
            score.dimension: score.score
            for score in scorecard.dimension_scores
            if score.score is not None and score.dimension in {"problem_understanding", "correctness"}
        }
        low_dimension_count = sum(
            1
            for score in scorecard.dimension_scores
            if score.score is not None and score.score < 3.0
        )
        scorecard.recommendation = build_recommendation(
            scorecard.overall_average,
            low_dimension_count=low_dimension_count,
            critical_dimension_scores=critical_dimensions,
        )
        strongest_text = ", ".join(strongest_dimensions[:2]) if strongest_dimensions else "balanced rubric results"
        weakest_text = ", ".join(weakest_dimensions[:2]) if weakest_dimensions else "no major rubric risks"
        scorecard.recommendation_rationale = (
            f"Recommendation is based on an overall rubric average of {scorecard.overall_average:.2f}, "
            f"strong signals in {strongest_text}, and recruiter caution around {weakest_text}."
        )
        scorecard.recommendation_blocked_reason = ""
        scorecard.generated_at = utc_now()
    else:
        scorecard.recommendation = None
        if session.status != SessionStatus.COMPLETED:
            scorecard.recommendation_blocked_reason = "Interview session has not completed yet."
        elif len(complete_attempts) != expected_questions:
            scorecard.recommendation_blocked_reason = (
                f"Only {len(complete_attempts)} of {expected_questions} primary question rubric(s) are fully scored."
            )
        else:
            scorecard.recommendation_blocked_reason = "Rubric scoring is incomplete."
        scorecard.recommendation_rationale = ""

    scorecard.summary = build_scorecard_summary(
        session,
        scorecard,
        strongest_dimensions,
        weakest_dimensions,
    )

    session.scorecard = scorecard
    interview_store.upsert_session(session)
    return scorecard


def build_attempt_timer_snapshot(attempt: QuestionAttempt | None) -> dict | None:
    if attempt is None:
        return None

    now = utc_now()
    remaining_seconds = None
    if attempt.expires_at is not None:
        remaining_seconds = max(0, int((attempt.expires_at - now).total_seconds()))

    return {
        "state": attempt.state.value,
        "started_at": attempt.started_at.isoformat() if attempt.started_at else None,
        "expires_at": attempt.expires_at.isoformat() if attempt.expires_at else None,
        "warning_threshold_seconds": attempt.warning_threshold_seconds,
        "remaining_seconds": remaining_seconds,
        "expired": attempt.state == QuestionState.EXPIRED,
        "locked": attempt.state in {QuestionState.EXPIRED, QuestionState.SCORED, QuestionState.ADVANCED},
        "lock_reason": attempt.lock_reason,
    }


def mark_attempt_warning(session: InterviewSession, attempt: QuestionAttempt, *, reason: str) -> None:
    if attempt.state != QuestionState.ACTIVE:
        return

    attempt.state = QuestionState.WARNING
    attempt.warning_emitted_at = attempt.warning_emitted_at or utc_now()
    interview_store.record_event(session.id, "timer_warning", reason)
    interview_store.upsert_session(session)


def mark_attempt_expired(session: InterviewSession, attempt: QuestionAttempt, *, reason: str) -> None:
    if attempt.state == QuestionState.EXPIRED:
        return

    now = utc_now()
    attempt.state = QuestionState.EXPIRED
    attempt.expired_at = attempt.expired_at or now
    attempt.locked_at = attempt.locked_at or now
    attempt.lock_reason = "time_cap_reached"
    attempt.recommended_next_step = NextStep.NEXT_QUESTION.value
    if not attempt.evaluation_summary:
        attempt.evaluation_summary = "Answer window expired before the question was evaluated."
    session.status = SessionStatus.REVIEW_PENDING
    interview_store.record_event(session.id, "timer_expired", reason)
    interview_store.upsert_session(session)
    refresh_scorecard(session)


def sync_attempt_timer(
    session: InterviewSession,
    attempt: QuestionAttempt | None,
    *,
    allow_warning_transition: bool = True,
) -> QuestionAttempt | None:
    if attempt is None or attempt.time_cap_seconds is None or attempt.expires_at is None:
        return attempt

    if attempt.state in {QuestionState.SCORED, QuestionState.ADVANCED, QuestionState.EXPIRED}:
        return attempt

    remaining_seconds = int((attempt.expires_at - utc_now()).total_seconds())
    if remaining_seconds <= 0:
        mark_attempt_expired(session, attempt, reason=f"Question {attempt.question_id} reached its hard time cap.")
        return attempt

    if (
        allow_warning_transition
        and attempt.state == QuestionState.ACTIVE
        and remaining_seconds <= attempt.warning_threshold_seconds
    ):
        mark_attempt_warning(
            session,
            attempt,
            reason=f"Question {attempt.question_id} entered warning with {remaining_seconds} second(s) remaining.",
        )

    return attempt


def require_active_attempt_for_answer(session: InterviewSession) -> QuestionAttempt:
    attempt = get_current_attempt(session)
    if attempt is None:
        raise HTTPException(status_code=400, detail="No active question exists for this session.")

    sync_attempt_timer(session, attempt)
    if attempt.state == QuestionState.EXPIRED:
        raise HTTPException(status_code=409, detail="The current question has expired and is locked.")

    return attempt


def apply_guardrails(session: InterviewSession, user_text: str):
    policy = get_interview_policy(session.mode)
    prior_user_texts = collect_prior_user_texts(session)
    return evaluate_candidate_message(user_text, policy=policy, prior_user_texts=prior_user_texts)


def generate_interview_question(
    prompt: str,
    fallback: str,
    *,
    mode: InterviewMode = InterviewMode.HIRING,
) -> str:
    try:
        return generate_ai_text(prompt, mode=mode)
    except Exception as e:
        print(f"LLM error: {e}")
        return fallback


def generate_ai_text(user_prompt: str, *, mode: InterviewMode = InterviewMode.HIRING) -> str:
    policy = get_interview_policy(mode)
    response = completion(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": build_system_prompt(policy)},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.replace("*", "").strip()


def persist_answer_evaluation(session: InterviewSession, attempt: QuestionAttempt, question: Question, user_text: str):
    evaluation = evaluate_answer(question, user_text, source_type=attempt.source_type)
    attempt.user_text = user_text
    attempt.submitted_at = utc_now()
    attempt.state = QuestionState.SCORED
    attempt.evaluation_summary = evaluation.summary
    attempt.recommended_next_step = evaluation.next_step.value
    attempt.expected_concepts_hit = evaluation.concept_hits
    attempt.missing_concepts = evaluation.missing_concepts
    attempt.selected_follow_up_id = evaluation.selected_follow_up_id
    attempt.scores = evaluation.dimension_scores
    attempt.notes.extend(evaluation.evidence)
    session.last_user_text = user_text
    session.status = SessionStatus.REVIEW_PENDING

    ai_text = build_review_acknowledgement(evaluation)
    attempt.ai_text = ai_text
    session.last_ai_text = ai_text
    interview_store.record_event(
        session.id,
        "answer_evaluated",
        f"{attempt.question_id}: {evaluation.summary}",
    )
    interview_store.upsert_session(session)
    refresh_scorecard(session)
    return evaluation


def advance_session_engine(session: InterviewSession, *, reason: str) -> tuple[str, dict]:
    current_attempt = get_current_attempt(session)
    if current_attempt is None:
        raise HTTPException(status_code=400, detail="No active question exists for this session.")

    sync_attempt_timer(session, current_attempt)
    if current_attempt.state not in {QuestionState.SCORED, QuestionState.EXPIRED, QuestionState.ADVANCED}:
        raise HTTPException(status_code=400, detail="Current question must be evaluated before advancing.")

    question = get_question_by_attempt(current_attempt)
    current_attempt.state = QuestionState.ADVANCED
    next_step = NextStep(current_attempt.recommended_next_step or NextStep.NEXT_QUESTION.value)

    if next_step == NextStep.FOLLOW_UP:
        follow_up = get_follow_up_by_id(question, current_attempt.selected_follow_up_id)
        if follow_up is None:
            next_step = NextStep.NEXT_QUESTION
        else:
            ai_text = build_interviewer_transition(
                next_step=next_step,
                follow_up=follow_up,
                mode=session.mode,
            )
            create_session_attempt(
                session,
                question_id=question.id,
                question_title=f"{question.title} follow-up",
                prompt=follow_up.prompt,
                time_cap_seconds=max(120, min(question.time_cap_seconds, 240)),
                source_type="follow_up",
                selected_follow_up_id=follow_up.id,
            )
            session.status = SessionStatus.IN_PROGRESS
            session.last_ai_text = ai_text
            interview_store.record_event(session.id, "question_advanced", reason)
            interview_store.upsert_session(session)
            return ai_text, {"next_step": next_step.value}

    next_question_index = count_primary_attempts(session)
    next_question = get_question_for_session_index(session, next_question_index)
    if next_question is None:
        session.status = SessionStatus.COMPLETED
        session.completed_at = utc_now()
        session.last_ai_text = "This interview session is complete. A recruiter scorecard can now be generated."
        interview_store.record_event(session.id, "session_completed", reason)
        interview_store.upsert_session(session)
        refresh_scorecard(session)
        return session.last_ai_text, {"next_step": NextStep.COMPLETE.value}

    ai_text = build_interviewer_transition(
        next_step=NextStep.NEXT_QUESTION,
        next_question=next_question,
        mode=session.mode,
    )
    create_session_attempt(
        session,
        question_id=next_question.id,
        question_title=next_question.title,
        prompt=next_question.prompt,
        time_cap_seconds=next_question.time_cap_seconds,
        source_type="question",
    )
    session.status = SessionStatus.IN_PROGRESS
    session.last_ai_text = ai_text
    interview_store.record_event(session.id, "question_advanced", reason)
    interview_store.upsert_session(session)
    return ai_text, {"next_step": NextStep.NEXT_QUESTION.value}


def save_response_audio(ai_text: str) -> None:
    tts = gTTS(text=ai_text, lang="en")
    tts.save("response.mp3")


def convert_upload_to_wav(file: UploadFile) -> str:
    original_ext = (file.filename or "audio.m4a").rsplit(".", 1)[-1].lower()
    input_filename = f"input_audio.{original_ext}"
    wav_filename = "converted_input.wav"

    with open(input_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    sound = AudioSegment.from_file(input_filename)
    sound.export(wav_filename, format="wav")
    return wav_filename


def get_whisper_model():
    global whisper_model

    if whisper_model is None:
        print("Loading Whisper model...")
        import whisper

        whisper_model = whisper.load_model("base")
        print("Whisper model ready.")

    return whisper_model


def transcribe_wav(wav_filename: str) -> str:
    result = get_whisper_model().transcribe(wav_filename)
    return result["text"].strip()


@app.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribes uploaded audio with Whisper and returns the text."""
    print(f"TRANSCRIBE: Processing file: {file.filename}")

    try:
        wav_filename = convert_upload_to_wav(file)
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return {"error": f"FFmpeg failed: {str(e)}", "user_text": ""}

    try:
        user_text = transcribe_wav(wav_filename)
        update_legacy_transcript(user_text=user_text)
        print(f"TRANSCRIBE: User said: {user_text}")
        return {"user_text": user_text}
    except Exception as e:
        print(f"Whisper STT error: {e}")
        return {"user_text": ""}


class AnswerPayload(BaseModel):
    user_text: str


class SessionCreatePayload(BaseModel):
    mode: InterviewMode = InterviewMode.HIRING
    template_id: str | None = None
    session_id: str | None = None


class SessionAnswerPayload(BaseModel):
    user_text: str


class SessionAdvancePayload(BaseModel):
    reason: str = "manual_advance"


class SessionTimerEventPayload(BaseModel):
    event: str = "sync"


@app.post("/sessions")
async def create_session(payload: SessionCreatePayload):
    template_id = payload.template_id or default_template_id_for_mode(payload.mode)
    if template_id not in interview_store.templates:
        raise HTTPException(status_code=404, detail=f"Unknown interview template: {template_id}")

    session = interview_store.create_session(
        template_id=template_id,
        mode=payload.mode,
        session_id=payload.session_id,
        status=SessionStatus.READY,
    )
    interview_store.record_event(session.id, "session_created", "Session created via API.")
    return build_session_payload(session)


@app.post("/sessions/{session_id}/start")
async def start_session(session_id: str):
    session = require_session_or_404(session_id)

    if session.status not in {SessionStatus.DRAFT, SessionStatus.READY}:
        return {
            "error": f"Session cannot start from status {session.status.value}.",
            **build_session_payload(session),
        }

    opening_question = get_question_for_session_index(session, 0)
    if opening_question is None:
        raise HTTPException(status_code=400, detail="Selected template does not contain any interview questions.")

    opening_prompt = opening_question.prompt

    session.status = SessionStatus.IN_PROGRESS
    session.started_at = session.started_at or utc_now()
    create_session_attempt(
        session,
        question_id=opening_question.id,
        question_title=opening_question.title,
        prompt=opening_prompt,
        time_cap_seconds=opening_question.time_cap_seconds,
    )
    interview_store.record_event(session.id, "session_started", "Opening interview question issued.")
    interview_store.upsert_session(session)
    save_response_audio(opening_prompt)
    return {"status": "ok", "ai_text": opening_prompt, **build_session_payload(session)}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    session = require_session_or_404(session_id)
    sync_attempt_timer(session, get_current_attempt(session))
    return build_session_payload(session)


@app.post("/sessions/{session_id}/answers")
async def submit_session_answer(session_id: str, payload: SessionAnswerPayload):
    session = require_session_or_404(session_id)

    if session.status != SessionStatus.IN_PROGRESS:
        return {
            "error": f"Answers are not accepted while session status is {session.status.value}.",
            **build_session_payload(session),
        }

    try:
        attempt = require_active_attempt_for_answer(session)
    except HTTPException as exc:
        if exc.status_code == 409:
            return {"error": exc.detail, **build_session_payload(session)}
        raise

    if attempt.state in {QuestionState.SUBMITTED, QuestionState.SCORED, QuestionState.ADVANCED}:
        return {"error": "The current question has already been answered.", **build_session_payload(session)}

    user_text = payload.user_text.strip()
    question = get_question_by_attempt(attempt)
    guardrail = apply_guardrails(session, user_text)
    if guardrail is not None:
        attempt.notes.append(guardrail.note)
        attempt.user_text = user_text
        attempt.ai_text = guardrail.response_text
        session.last_user_text = user_text
        session.last_ai_text = guardrail.response_text
        interview_store.record_event(
            session.id,
            f"guardrail_{guardrail.category.value}",
            guardrail.note,
        )
        interview_store.upsert_session(session)
        save_response_audio(guardrail.response_text)
        return {"status": "guardrail_triggered", "ai_text": guardrail.response_text, **build_session_payload(session)}

    evaluation = persist_answer_evaluation(session, attempt, question, user_text)
    ai_text = session.last_ai_text
    save_response_audio(ai_text)
    return {
        "status": "ok",
        "ai_text": ai_text,
        "engine_evaluation": evaluation.to_dict(),
        **build_session_payload(session),
    }


@app.post("/sessions/{session_id}/timer-events")
async def record_session_timer_event(session_id: str, payload: SessionTimerEventPayload):
    session = require_session_or_404(session_id)
    attempt = get_current_attempt(session)
    if attempt is None:
        return {"status": "idle", **build_session_payload(session)}

    if payload.event == "warning":
        sync_attempt_timer(session, attempt, allow_warning_transition=True)
        if attempt.state == QuestionState.ACTIVE:
            mark_attempt_warning(session, attempt, reason=f"Client warning event for {attempt.question_id}.")
        return {
            "status": "warning" if attempt.state == QuestionState.WARNING else "ok",
            "ai_text": "You are close to the time limit for this question.",
            **build_session_payload(session),
        }

    if payload.event == "expire":
        mark_attempt_expired(session, attempt, reason=f"Client expiry event for {attempt.question_id}.")
        return {
            "status": "expired",
            "ai_text": "Time is up for this question. Your answer window is now closed.",
            **build_session_payload(session),
        }

    sync_attempt_timer(session, attempt)
    return {
        "status": "expired" if attempt.state == QuestionState.EXPIRED else "ok",
        "ai_text": (
            "Time is up for this question. Your answer window is now closed."
            if attempt.state == QuestionState.EXPIRED
            else "Timer synchronized."
        ),
        **build_session_payload(session),
    }


@app.post("/sessions/{session_id}/advance")
async def advance_session(session_id: str, payload: SessionAdvancePayload):
    session = require_session_or_404(session_id)
    current_attempt = get_current_attempt(session)
    sync_attempt_timer(session, current_attempt)

    if session.status not in {SessionStatus.IN_PROGRESS, SessionStatus.REVIEW_PENDING}:
        return {
            "error": f"Session cannot advance from status {session.status.value}.",
            **build_session_payload(session),
        }

    if current_attempt is not None and current_attempt.state not in {
        QuestionState.SCORED,
        QuestionState.EXPIRED,
        QuestionState.ADVANCED,
    }:
        return {
            "error": "Current question must be evaluated before advancing.",
            **build_session_payload(session),
        }

    ai_text, engine_decision = advance_session_engine(session, reason=payload.reason)
    save_response_audio(ai_text)
    status = "completed" if engine_decision["next_step"] == NextStep.COMPLETE.value else "ok"
    return {"status": status, "ai_text": ai_text, "engine_decision": engine_decision, **build_session_payload(session)}


@app.get("/sessions/{session_id}/scorecard")
def get_session_scorecard(session_id: str):
    session = require_session_or_404(session_id)
    scorecard = refresh_scorecard(session)
    return {
        "session_id": session.id,
        "status": session.status.value,
        "template_id": session.template_id,
        "scorecard": scorecard.model_dump(mode="json"),
        "question_timeline": [entry.model_dump(mode="json") for entry in scorecard.question_summaries],
        "session_events": [event.model_dump(mode="json") for event in session.events],
    }


@app.post("/submit-answer")
async def submit_answer(payload: AnswerPayload):
    """Receives user text, runs the configured LLM, and generates TTS audio."""
    user_text = payload.user_text.strip()
    legacy_session = get_legacy_session()
    current_attempt = get_current_attempt(legacy_session)
    if current_attempt is None:
        opening_question = get_question_for_session_index(legacy_session, 0)
        if opening_question is not None:
            create_session_attempt(
                legacy_session,
                question_id=opening_question.id,
                question_title=opening_question.title,
                prompt=opening_question.prompt,
                time_cap_seconds=opening_question.time_cap_seconds,
                source_type="question",
            )
            interview_store.upsert_session(legacy_session)
            current_attempt = get_current_attempt(legacy_session)

    if current_attempt is None:
        ai_text = "I couldn't find an active interview question. Please restart the interview."
        save_response_audio(ai_text)
        return {"status": "error", "ai_text": ai_text}

    sync_attempt_timer(legacy_session, current_attempt)
    if current_attempt.state == QuestionState.EXPIRED:
        ai_text, _ = advance_session_engine(legacy_session, reason="legacy_submit_answer_after_expiry")
        update_legacy_transcript(ai_text=ai_text)
        save_response_audio(ai_text)
        return {"status": "expired", "ai_text": ai_text}

    guardrail = evaluate_candidate_message(
        user_text,
        policy=get_interview_policy(legacy_session.mode),
        prior_user_texts=collect_prior_user_texts(legacy_session),
    )
    update_legacy_transcript(user_text=user_text)
    print(f"SUBMIT: User text: {user_text}")

    if guardrail is not None:
        interview_store.record_event(
            legacy_session.id,
            f"guardrail_{guardrail.category.value}",
            guardrail.note,
        )
        ai_text = guardrail.response_text
    elif user_text:
        question = get_question_by_attempt(current_attempt)
        persist_answer_evaluation(legacy_session, current_attempt, question, user_text)
        ai_text, _ = advance_session_engine(legacy_session, reason="legacy_submit_answer")
    else:
        ai_text = "I couldn't hear you clearly. Could you please repeat your answer?"

    update_legacy_transcript(ai_text=ai_text)
    print(f"SUBMIT: AI Response: {ai_text}")

    save_response_audio(ai_text)
    return {"status": "ok", "ai_text": ai_text}


@app.post("/start-interview")
async def start_interview():
    print("START INTERVIEW: Generating opening question...")

    try:
        session = get_legacy_session()
        opening_question = get_question_for_session_index(session, 0)
        if opening_question is None:
            raise ValueError("No questions found for the legacy interview template.")
        ai_text = opening_question.prompt
        if not get_current_attempt(session):
            create_session_attempt(
                session,
                question_id=opening_question.id,
                question_title=opening_question.title,
                prompt=opening_question.prompt,
                time_cap_seconds=opening_question.time_cap_seconds,
                source_type="question",
            )
    except Exception as e:
        print(f"Question bank error: {e}")
        session = get_legacy_session()
        ai_text = (
            "Let's start with a DSA warm-up: explain how you would find duplicate values in an integer array "
            "and what trade-offs you would consider."
        )

    session.status = SessionStatus.IN_PROGRESS
    interview_store.upsert_session(session)
    update_legacy_transcript(ai_text=ai_text)
    print(f"Opening question: {ai_text}")
    save_response_audio(ai_text)
    return {"status": "ok", "ai_text": ai_text}


@app.get("/get-audio")
def get_audio():
    return FileResponse("response.mp3", media_type="audio/mpeg")


@app.get("/get-transcript")
def get_transcript():
    session = get_legacy_session()
    sync_attempt_timer(session, get_current_attempt(session))
    return {"user_text": session.last_user_text}


@app.get("/get-ai-transcript")
def get_ai_transcript():
    session = get_legacy_session()
    sync_attempt_timer(session, get_current_attempt(session))
    return {"ai_text": session.last_ai_text}


@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """Legacy endpoint: transcribes audio, runs the configured LLM, and returns TTS."""
    print(f"PROCESS-AUDIO: {file.filename}")

    try:
        wav_filename = convert_upload_to_wav(file)
    except Exception as e:
        return {"error": f"FFmpeg failed: {str(e)}"}

    user_text = ""
    legacy_session = get_legacy_session()
    guardrail = None
    try:
        user_text = transcribe_wav(wav_filename)
        guardrail = evaluate_candidate_message(
            user_text,
            policy=get_interview_policy(legacy_session.mode),
            prior_user_texts=collect_prior_user_texts(legacy_session),
        )
        update_legacy_transcript(user_text=user_text)
        print(f"User said: {user_text}")
    except Exception as e:
        print(f"Whisper STT error: {e}")

    if user_text:
        if guardrail is not None:
            interview_store.record_event(
                legacy_session.id,
                f"guardrail_{guardrail.category.value}",
                guardrail.note,
            )
            ai_text = guardrail.response_text
            update_legacy_transcript(ai_text=ai_text)
            print(f"AI Response: {ai_text}")
            save_response_audio(ai_text)
            return FileResponse("response.mp3", media_type="audio/mpeg")

        current_attempt = get_current_attempt(legacy_session)
        if current_attempt is None:
            opening_question = get_question_for_session_index(legacy_session, 0)
            if opening_question is not None:
                create_session_attempt(
                    legacy_session,
                    question_id=opening_question.id,
                    question_title=opening_question.title,
                    prompt=opening_question.prompt,
                    time_cap_seconds=opening_question.time_cap_seconds,
                    source_type="question",
                )
                interview_store.upsert_session(legacy_session)
                current_attempt = get_current_attempt(legacy_session)

        if current_attempt is None:
            ai_text = "I couldn't find an active interview question."
        elif sync_attempt_timer(legacy_session, current_attempt) and current_attempt.state == QuestionState.EXPIRED:
            ai_text, _ = advance_session_engine(legacy_session, reason="legacy_process_audio_after_expiry")
        else:
            question = get_question_by_attempt(current_attempt)
            persist_answer_evaluation(legacy_session, current_attempt, question, user_text)
            ai_text, _ = advance_session_engine(legacy_session, reason="legacy_process_audio")
    else:
        ai_text = "I couldn't hear you."

    update_legacy_transcript(ai_text=ai_text)
    print(f"AI Response: {ai_text}")

    save_response_audio(ai_text)
    return FileResponse("response.mp3", media_type="audio/mpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
