import os
import shutil

import static_ffmpeg

static_ffmpeg.add_paths()

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from gtts import gTTS
from litellm import completion
from pydantic import BaseModel
from pydub import AudioSegment

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash")
INTERVIEWER_SYSTEM_PROMPT = (
    "You are InterVo, a professional mathematics interviewer conducting an "
    "admission interview for SST. Your goal is to help candidates clarify "
    "their math concepts."
)

# Set ffmpeg paths required by pydub.
AudioSegment.converter = shutil.which("ffmpeg")
AudioSegment.ffmpeg = shutil.which("ffmpeg")
AudioSegment.ffprobe = shutil.which("ffprobe")

app = FastAPI()

# In-memory transcript state.
last_user_text: str = ""
last_ai_text: str = ""
whisper_model = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_ai_text(user_prompt: str) -> str:
    response = completion(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": INTERVIEWER_SYSTEM_PROMPT},
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
    global last_user_text
    print(f"TRANSCRIBE: Processing file: {file.filename}")

    try:
        wav_filename = convert_upload_to_wav(file)
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return {"error": f"FFmpeg failed: {str(e)}", "user_text": ""}

    try:
        user_text = transcribe_wav(wav_filename)
        last_user_text = user_text
        print(f"TRANSCRIBE: User said: {user_text}")
        return {"user_text": user_text}
    except Exception as e:
        print(f"Whisper STT error: {e}")
        return {"user_text": ""}


class AnswerPayload(BaseModel):
    user_text: str


@app.post("/submit-answer")
async def submit_answer(payload: AnswerPayload):
    """Receives user text, runs the configured LLM, and generates TTS audio."""
    global last_user_text, last_ai_text

    user_text = payload.user_text.strip()
    last_user_text = user_text
    print(f"SUBMIT: User text: {user_text}")

    if user_text:
        try:
            ai_text = generate_ai_text(
                f"The candidate said: {user_text}. "
                "Respond naturally as InterVo: briefly evaluate or acknowledge "
                "their answer, provide a tiny hint or correction if their math "
                "concept was slightly off, and ask the next math-related question. "
                "Keep your response strictly to 2-3 sentences."
            )
        except Exception as e:
            print(f"LLM error: {e}")
            ai_text = "Thank you for your answer. Let's move on to the next question."
    else:
        ai_text = "I couldn't hear you clearly. Could you please repeat your answer?"

    last_ai_text = ai_text
    print(f"SUBMIT: AI Response: {ai_text}")

    save_response_audio(ai_text)
    return {"status": "ok", "ai_text": ai_text}


@app.post("/start-interview")
async def start_interview():
    global last_ai_text
    print("START INTERVIEW: Generating opening question...")

    try:
        ai_text = generate_ai_text(
            "Start the interview with a warm greeting, introduce yourself as "
            "InterVo, and ask the candidate a thought-provoking introductory "
            "math question. Keep your response strictly to 2-3 sentences."
        )
    except Exception as e:
        print(f"LLM error: {e}")
        ai_text = (
            "Hello! I am InterVo, your mathematics interviewer. Let's start "
            "with a warm-up: Can you explain an interesting mathematical "
            "concept you recently learned?"
        )

    last_ai_text = ai_text
    print(f"Opening question: {ai_text}")
    save_response_audio(ai_text)
    return {"status": "ok", "ai_text": ai_text}


@app.get("/get-audio")
def get_audio():
    return FileResponse("response.mp3", media_type="audio/mpeg")


@app.get("/get-transcript")
def get_transcript():
    return {"user_text": last_user_text}


@app.get("/get-ai-transcript")
def get_ai_transcript():
    return {"ai_text": last_ai_text}


@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """Legacy endpoint: transcribes audio, runs the configured LLM, and returns TTS."""
    global last_user_text, last_ai_text
    print(f"PROCESS-AUDIO: {file.filename}")

    try:
        wav_filename = convert_upload_to_wav(file)
    except Exception as e:
        return {"error": f"FFmpeg failed: {str(e)}"}

    user_text = ""
    try:
        user_text = transcribe_wav(wav_filename)
        last_user_text = user_text
        print(f"User said: {user_text}")
    except Exception as e:
        print(f"Whisper STT error: {e}")

    if user_text:
        try:
            ai_text = generate_ai_text(
                f"The user said: {user_text}. Reply in 1 brief sentence about math."
            )
        except Exception as e:
            print(f"LLM error: {e}")
            ai_text = "I am having trouble thinking."
    else:
        ai_text = "I couldn't hear you."

    last_ai_text = ai_text
    print(f"AI Response: {ai_text}")

    save_response_audio(ai_text)
    return FileResponse("response.mp3", media_type="audio/mpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
