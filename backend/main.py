import os
import shutil

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
    InMemoryInterviewStore,
    InterviewMode,
    InterviewSession,
    QuestionAttempt,
    QuestionState,
    SessionStatus,
    utc_now,
)
from interview_policy import build_system_prompt, evaluate_candidate_message, get_interview_policy
from litellm import completion
from pydantic import BaseModel
from pydub import AudioSegment
from question_bank import default_template_id_for_mode, seed_question_bank

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash")
DEFAULT_SESSION_TIME_CAP_SECONDS = 300
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
        attempt.state = QuestionState.SUBMITTED if user_text else QuestionState.READY
    if ai_text is not None:
        session.last_ai_text = ai_text
        attempt.ai_text = ai_text
        if ai_text:
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
) -> QuestionAttempt:
    attempt = QuestionAttempt(
        question_id=question_id,
        question_title=question_title,
        ai_text=prompt,
        state=QuestionState.ACTIVE,
        time_cap_seconds=time_cap_seconds,
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
    return build_session_payload(session)


@app.post("/sessions/{session_id}/answers")
async def submit_session_answer(session_id: str, payload: SessionAnswerPayload):
    session = require_session_or_404(session_id)

    if session.status != SessionStatus.IN_PROGRESS:
        return {
            "error": f"Answers are not accepted while session status is {session.status.value}.",
            **build_session_payload(session),
        }

    attempt = get_current_attempt(session)
    if attempt is None:
        return {"error": "No active question exists for this session.", **build_session_payload(session)}

    if attempt.state in {QuestionState.SUBMITTED, QuestionState.SCORED, QuestionState.ADVANCED}:
        return {"error": "The current question has already been answered.", **build_session_payload(session)}

    user_text = payload.user_text.strip()
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

    attempt.user_text = user_text
    attempt.submitted_at = utc_now()
    attempt.state = QuestionState.SUBMITTED
    session.last_user_text = user_text
    session.status = SessionStatus.REVIEW_PENDING

    ai_text = (
        "Answer captured. The interviewer will review this response and move to the next question when ready."
    )
    attempt.ai_text = ai_text
    session.last_ai_text = ai_text

    interview_store.record_event(session.id, "answer_submitted", f"Answer submitted for {attempt.question_id}.")
    interview_store.upsert_session(session)
    save_response_audio(ai_text)
    return {"status": "ok", "ai_text": ai_text, **build_session_payload(session)}


@app.post("/sessions/{session_id}/advance")
async def advance_session(session_id: str, payload: SessionAdvancePayload):
    session = require_session_or_404(session_id)
    current_attempt = get_current_attempt(session)

    if session.status not in {SessionStatus.IN_PROGRESS, SessionStatus.REVIEW_PENDING}:
        return {
            "error": f"Session cannot advance from status {session.status.value}.",
            **build_session_payload(session),
        }

    if current_attempt is not None and current_attempt.state not in {
        QuestionState.SUBMITTED,
        QuestionState.SCORED,
        QuestionState.ADVANCED,
    }:
        return {
            "error": "Current question must be submitted before advancing.",
            **build_session_payload(session),
        }

    if current_attempt is not None:
        current_attempt.state = QuestionState.ADVANCED

    next_question_index = len(session.attempts)
    next_question = get_question_for_session_index(session, next_question_index)
    if next_question is None:
        session.status = SessionStatus.COMPLETED
        session.completed_at = utc_now()
        interview_store.record_event(session.id, "session_completed", payload.reason)
        interview_store.upsert_session(session)
        return {
            "status": "completed",
            "ai_text": "This interview session is complete. A recruiter scorecard can now be generated.",
            **build_session_payload(session),
        }

    next_prompt = next_question.prompt
    create_session_attempt(
        session,
        question_id=next_question.id,
        question_title=next_question.title,
        prompt=next_prompt,
        time_cap_seconds=next_question.time_cap_seconds,
    )
    session.status = SessionStatus.IN_PROGRESS

    interview_store.record_event(session.id, "question_advanced", payload.reason)
    interview_store.upsert_session(session)
    save_response_audio(next_prompt)
    return {"status": "ok", "ai_text": next_prompt, **build_session_payload(session)}


@app.get("/sessions/{session_id}/scorecard")
def get_session_scorecard(session_id: str):
    session = require_session_or_404(session_id)
    return {
        "session_id": session.id,
        "status": session.status.value,
        "scorecard": session.scorecard.model_dump(mode="json"),
    }


@app.post("/submit-answer")
async def submit_answer(payload: AnswerPayload):
    """Receives user text, runs the configured LLM, and generates TTS audio."""
    user_text = payload.user_text.strip()
    legacy_session = get_legacy_session()
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
        try:
            ai_text = generate_ai_text(build_legacy_answer_prompt(user_text), mode=legacy_session.mode)
        except Exception as e:
            print(f"LLM error: {e}")
            ai_text = "Thank you for your answer. Let's move on to the next question."
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
    return {"user_text": session.last_user_text}


@app.get("/get-ai-transcript")
def get_ai_transcript():
    session = get_legacy_session()
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

        try:
            ai_text = generate_ai_text(build_legacy_answer_prompt(user_text), mode=legacy_session.mode)
        except Exception as e:
            print(f"LLM error: {e}")
            ai_text = "I am having trouble thinking."
    else:
        ai_text = "I couldn't hear you."

    update_legacy_transcript(ai_text=ai_text)
    print(f"AI Response: {ai_text}")

    save_response_audio(ai_text)
    return FileResponse("response.mp3", media_type="audio/mpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
