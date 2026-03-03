import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import google.generativeai as genai
from gtts import gTTS
from pydub import AudioSegment
from dotenv import load_dotenv

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('models/gemini-2.5-flash')

# Set ffmpeg paths (required by pydub)
AudioSegment.converter = shutil.which("ffmpeg")
AudioSegment.ffmpeg = shutil.which("ffmpeg")
AudioSegment.ffprobe = shutil.which("ffprobe")

# Load Whisper STT model once at startup
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model ready.")

app = FastAPI()

# ─── In-memory transcript state ───────────────────────────────────────────────
last_user_text: str = ""
last_ai_text:   str = ""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 1. Transcribe only (Whisper, no Gemini) ──────────────────────────────────
@app.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribes uploaded audio with Whisper and returns the text."""
    global last_user_text
    print(f"TRANSCRIBE: Processing file: {file.filename}")

    original_ext = (file.filename or "audio.m4a").rsplit(".", 1)[-1].lower()
    input_filename = f"input_audio.{original_ext}"
    wav_filename = "converted_input.wav"

    with open(input_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        sound = AudioSegment.from_file(input_filename)
        sound.export(wav_filename, format="wav")
    except Exception as e:
        print(f"❌ FFMPEG ERROR: {e}")
        return {"error": f"FFmpeg failed: {str(e)}", "user_text": ""}

    try:
        result = whisper_model.transcribe(wav_filename)
        user_text = result["text"].strip()
        last_user_text = user_text
        print(f"TRANSCRIBE: User said: {user_text}")
        return {"user_text": user_text}
    except Exception as e:
        print(f"❌ Whisper STT Error: {e}")
        return {"user_text": ""}


# ─── 2. Submit answer (Gemini + TTS from text) ────────────────────────────────
class AnswerPayload(BaseModel):
    user_text: str

@app.post("/submit-answer")
async def submit_answer(payload: AnswerPayload):
    """Receives (possibly edited) user text, runs Gemini, generates TTS audio."""
    global last_user_text, last_ai_text

    user_text = payload.user_text.strip()
    last_user_text = user_text
    print(f"SUBMIT: User text: {user_text}")

    if user_text:
        try:
            response = model.generate_content(
                f"You are a professional interviewer conducting a Mathematics admission interview for SST. "
                f"The candidate said: {user_text}. "
                f"Respond naturally as an interviewer — acknowledge their answer briefly and ask the next question. "
                f"Keep it to 2-3 sentences."
            )
            ai_text = response.text.replace("*", "")
        except Exception as e:
            print(f"❌ Gemini Error: {e}")
            ai_text = "Thank you for your answer. Let's move on to the next question."
    else:
        ai_text = "I couldn't hear you clearly. Could you please repeat your answer?"

    last_ai_text = ai_text
    print(f"SUBMIT: AI Response: {ai_text}")

    tts = gTTS(text=ai_text, lang='en')
    tts.save("response.mp3")
    return {"status": "ok", "ai_text": ai_text}


# ─── 3. Start interview ────────────────────────────────────────────────────────
@app.post("/start-interview")
async def start_interview():
    global last_ai_text
    print("START INTERVIEW: Generating opening question...")
    try:
        response = model.generate_content(
            "You are a professional interviewer conducting a Mathematics admission interview for SST. "
            "Start the interview with a warm greeting and ask the candidate their first question. "
            "Keep it to 2-3 sentences."
        )
        ai_text = response.text.replace("*", "")
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        ai_text = "Hello! Welcome to your mathematics interview. Let's start with a warm-up question: Can you tell me a little about yourself and your interest in mathematics?"

    last_ai_text = ai_text
    print(f"Opening question: {ai_text}")
    tts = gTTS(text=ai_text, lang='en')
    tts.save("response.mp3")
    return {"status": "ok", "ai_text": ai_text}


# ─── 4. Serve audio ────────────────────────────────────────────────────────────
@app.get("/get-audio")
def get_audio():
    return FileResponse("response.mp3", media_type="audio/mpeg")


# ─── 5. Transcript accessors ──────────────────────────────────────────────────
@app.get("/get-transcript")
def get_transcript():
    return {"user_text": last_user_text}

@app.get("/get-ai-transcript")
def get_ai_transcript():
    return {"ai_text": last_ai_text}


# ─── Legacy endpoint (kept for compatibility) ─────────────────────────────────
@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """Legacy combined endpoint — transcribes + Gemini + TTS in one call."""
    global last_user_text, last_ai_text
    print(f"1. PROCESS-AUDIO: {file.filename}")

    original_ext = (file.filename or "audio.m4a").rsplit(".", 1)[-1].lower()
    input_filename = f"input_audio.{original_ext}"
    wav_filename = "converted_input.wav"

    with open(input_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        sound = AudioSegment.from_file(input_filename)
        sound.export(wav_filename, format="wav")
    except Exception as e:
        return {"error": f"FFmpeg failed: {str(e)}"}

    user_text = ""
    try:
        result = whisper_model.transcribe(wav_filename)
        user_text = result["text"].strip()
        last_user_text = user_text
        print(f"User said: {user_text}")
    except Exception as e:
        print(f"❌ Whisper STT Error: {e}")

    if user_text:
        try:
            response = model.generate_content(f"User said: {user_text}. Reply in 1 sentence.")
            ai_text = response.text.replace("*", "")
        except Exception as e:
            ai_text = "I am having trouble thinking."
    else:
        ai_text = "I couldn't hear you."

    last_ai_text = ai_text
    print(f"AI Response: {ai_text}")

    tts = gTTS(text=ai_text, lang='en')
    tts.save("response.mp3")
    return FileResponse("response.mp3", media_type="audio/mpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)