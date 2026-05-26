import asyncio

from interview_domain import InterviewMode, QuestionState, SessionStatus
from main import (
    SessionAnswerPayload,
    create_session_attempt,
    default_template_id_for_mode,
    get_question_for_session_index,
    interview_store,
    is_empty_or_silence,
    submit_session_answer,
)


def build_active_session(session_id: str):
    session = interview_store.create_session(
        template_id=default_template_id_for_mode(InterviewMode.HIRING),
        mode=InterviewMode.HIRING,
        session_id=session_id,
        status=SessionStatus.IN_PROGRESS,
    )
    question = get_question_for_session_index(session, 0)
    assert question is not None
    create_session_attempt(
        session,
        question_id=question.id,
        question_title=question.title,
        prompt=question.prompt,
        time_cap_seconds=question.time_cap_seconds,
    )
    interview_store.upsert_session(session)
    return session


async def main():
    assert is_empty_or_silence("")
    assert is_empty_or_silence("   um uh   ")
    assert not is_empty_or_silence("I would use a hash map and explain the complexity.")

    empty_session = build_active_session("reliability-empty-answer")
    empty_response = await submit_session_answer(
        empty_session.id,
        SessionAnswerPayload(user_text="   um   "),
    )
    empty_attempt = empty_response["current_attempt"]
    assert empty_response["status"] == "empty_answer"
    assert empty_attempt["state"] == QuestionState.ACTIVE.value
    assert empty_attempt["user_text"] == ""

    stale_session = build_active_session("reliability-stale-attempt")
    stale_response = await submit_session_answer(
        stale_session.id,
        SessionAnswerPayload(user_text="I would use a hash map.", attempt_id="old-attempt-id"),
    )
    assert stale_response["status"] == "stale_attempt"
    assert stale_response["current_attempt"]["state"] == QuestionState.ACTIVE.value

    print("reliability checks ok")


if __name__ == "__main__":
    asyncio.run(main())
