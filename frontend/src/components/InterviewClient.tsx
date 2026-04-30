"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  Bot,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  Mic,
  Play,
  RefreshCw,
  Send,
  Square,
  User,
} from "lucide-react";

const SERVER_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const TARGET_QUESTIONS = 6;

type Phase =
  | "idle"
  | "ai_speaking"
  | "ready"
  | "recording"
  | "transcribing"
  | "pending_submit"
  | "submitting";

interface Message {
  role: "ai" | "user";
  text: string;
}

const phaseLabels: Record<Phase, string> = {
  idle: "Ready",
  ai_speaking: "InterVo speaking",
  ready: "Candidate turn",
  recording: "Recording",
  transcribing: "Transcribing",
  pending_submit: "Review answer",
  submitting: "Evaluating",
};

const focusAreas = ["Reasoning", "Clarity", "Core concepts", "Follow-up depth"];

function formatTime(seconds: number) {
  const mins = Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0");
  const secs = (seconds % 60).toString().padStart(2, "0");
  return `${mins}:${secs}`;
}

export default function InterviewClient() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [interviewStarted, setInterviewStarted] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [draftAnswer, setDraftAnswer] = useState("");
  const [error, setError] = useState("");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const userAnswerCount = messages.filter((message) => message.role === "user").length;
  const aiQuestionCount = messages.filter((message) => message.role === "ai").length;
  const progress = Math.min(100, Math.round((userAnswerCount / TARGET_QUESTIONS) * 100));
  const statusLabel = phaseLabels[phase];
  const isAiSpeaking = phase === "ai_speaking";
  const isRecording = phase === "recording";
  const isPendingSubmit = phase === "pending_submit";
  const isBusy = ["submitting", "transcribing", "ai_speaking"].includes(phase);

  const currentGuidance = useMemo(() => {
    if (!interviewStarted) return "Start a structured mock admission interview.";
    if (phase === "ai_speaking") return "Listen to the full question before answering.";
    if (phase === "recording") return "Answer aloud with steps, assumptions, and conclusion.";
    if (phase === "pending_submit") return "Clean the transcript before submitting it.";
    if (phase === "transcribing") return "Converting your voice into editable text.";
    if (phase === "submitting") return "Generating feedback and the next question.";
    return "Record your next answer when ready.";
  }, [interviewStarted, phase]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, draftAnswer, phase]);

  useEffect(() => {
    if (!interviewStarted) return;
    const timer = window.setInterval(() => {
      setElapsedSeconds((value) => value + 1);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [interviewStarted]);

  useEffect(() => {
    return () => {
      audioRef.current?.pause();
      mediaRecorderRef.current?.stream.getTracks().forEach((track) => track.stop());
    };
  }, []);

  function addMessage(role: "ai" | "user", text: string) {
    setMessages((previous) => [...previous, { role, text }]);
  }

  function setFailure(message: string) {
    setError(message);
    setPhase(interviewStarted ? "ready" : "idle");
  }

  async function beginInterview() {
    setError("");
    setPhase("submitting");

    try {
      const response = await fetch(`${SERVER_URL}/start-interview`, { method: "POST" });
      if (!response.ok) throw new Error(`Server returned ${response.status}`);

      const data = await response.json();
      setInterviewStarted(true);
      setElapsedSeconds(0);
      await playAiResponse(data.ai_text || null);
    } catch (err) {
      console.error(err);
      setFailure("Backend is not reachable. Start FastAPI on port 8000 and try again.");
    }
  }

  async function playAiResponse(aiTextHint: string | null) {
    setPhase("ai_speaking");

    try {
      let aiText = aiTextHint;
      if (!aiText) {
        const response = await fetch(`${SERVER_URL}/get-ai-transcript`);
        if (response.ok) {
          const data = await response.json();
          aiText = data.ai_text;
        }
      }

      audioRef.current?.pause();
      const audio = new Audio(`${SERVER_URL}/get-audio?rnd=${Date.now()}`);
      audioRef.current = audio;

      audio.onended = () => {
        addMessage("ai", aiText || "InterVo responded.");
        setPhase("ready");
      };

      await audio.play();
    } catch (err) {
      console.error("Playback error:", err);
      addMessage("ai", aiTextHint || "InterVo responded.");
      setPhase("ready");
    }
  }

  async function startRecording() {
    setError("");

    try {
      audioRef.current?.pause();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      const recorder = new MediaRecorder(stream, { mimeType });

      audioChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current = recorder;
      recorder.start(100);
      setPhase("recording");
    } catch (err) {
      console.error(err);
      setFailure("Microphone permission is required for the voice interview.");
    }
  }

  async function stopRecording() {
    if (phase !== "recording") return;

    const recorder = mediaRecorderRef.current;
    if (!recorder) return;

    setPhase("transcribing");
    recorder.onstop = async () => {
      const blob = new Blob(audioChunksRef.current, { type: recorder.mimeType });
      recorder.stream.getTracks().forEach((track) => track.stop());
      await transcribeAudio(blob);
    };
    recorder.stop();
  }

  async function transcribeAudio(blob: Blob) {
    try {
      const formData = new FormData();
      const ext = blob.type.includes("ogg") ? "ogg" : blob.type.includes("mp4") ? "mp4" : "webm";
      formData.append("file", blob, `recording.${ext}`);

      const response = await fetch(`${SERVER_URL}/transcribe-audio`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error(`Server returned ${response.status}`);

      const data = await response.json();
      setDraftAnswer(data.user_text || "");
      setPhase("pending_submit");
    } catch (err) {
      console.error(err);
      setFailure("Transcription failed. Check the backend logs and record again.");
    }
  }

  async function submitAnswer() {
    const text = draftAnswer.trim();
    if (!text) return;

    setError("");
    setPhase("submitting");
    addMessage("user", text);
    setDraftAnswer("");

    try {
      const response = await fetch(`${SERVER_URL}/submit-answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_text: text }),
      });
      if (!response.ok) throw new Error(`Server returned ${response.status}`);

      const data = await response.json();
      await playAiResponse(data.ai_text || null);
    } catch (err) {
      console.error(err);
      setFailure("Could not submit the answer. Confirm Gemini and gTTS are configured.");
    }
  }

  function discardDraft() {
    setDraftAnswer("");
    setPhase("ready");
  }

  return (
    <div className="interview-shell">
      <header className="app-header">
        <div className="brand-block">
          <div className="brand-mark">
            <BrainCircuit size={21} />
          </div>
          <div>
            <p className="eyebrow">AI mathematics interviewer</p>
            <h1>InterVo</h1>
          </div>
        </div>

        <div className={`live-status live-status-${phase}`}>
          <span />
          {statusLabel}
        </div>

        <div className="session-time">
          <Clock3 size={16} />
          {formatTime(elapsedSeconds)}
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <AlertCircle size={17} />
          {error}
        </div>
      )}

      <main className="workspace">
        <section className="stage-panel">
          <div className="stage-header">
            <div>
              <p className="eyebrow">Live assessment room</p>
              <h2>Admission interview simulation</h2>
            </div>
            <div className="question-meter">
              <span>{userAnswerCount}/{TARGET_QUESTIONS}</span>
              answers
            </div>
          </div>

          <div className="progress-track" aria-label="Interview progress">
            <div style={{ width: `${progress}%` }} />
          </div>

          <div className="participant-grid">
            <article className={`person-card interviewer ${isAiSpeaking ? "speaking" : ""}`}>
              <div className="person-background" />
              <div className="avatar-ring">
                <Bot size={52} />
                {isAiSpeaking && (
                  <div className="voice-bars" aria-hidden="true">
                    <i />
                    <i />
                    <i />
                    <i />
                  </div>
                )}
              </div>
              <div className="person-footer">
                <div>
                  <strong>InterVo</strong>
                  <span>Adaptive math interviewer</span>
                </div>
                <Mic size={16} />
              </div>
            </article>

            <article className={`person-card candidate ${isRecording ? "recording" : ""}`}>
              <div className="person-background candidate-bg" />
              <div className="avatar-ring candidate-avatar">
                <User size={52} />
              </div>
              <div className="person-footer">
                <div>
                  <strong>Candidate</strong>
                  <span>{isRecording ? "Answering now" : "Waiting turn"}</span>
                </div>
                <Mic size={16} />
              </div>
            </article>
          </div>

          <div className="control-strip">
            {!interviewStarted ? (
              <button className="primary-action" onClick={beginInterview} disabled={phase === "submitting"}>
                {phase === "submitting" ? <RefreshCw className="spin" size={18} /> : <Play size={18} />}
                {phase === "submitting" ? "Connecting" : "Begin interview"}
              </button>
            ) : isPendingSubmit ? (
              <div className="review-reminder">
                <CheckCircle2 size={18} />
                Review the transcript before sending it to InterVo.
              </div>
            ) : (
              <button
                className={`record-action ${isRecording ? "recording" : ""}`}
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isBusy || isPendingSubmit}
                title={isRecording ? "Stop and transcribe" : "Start recording"}
              >
                {isRecording ? <Square size={22} fill="currentColor" /> : <Mic size={24} />}
              </button>
            )}
            <p>{currentGuidance}</p>
          </div>
        </section>

        <aside className="insight-panel">
          <section className="metric-row">
            <div>
              <span>{aiQuestionCount}</span>
              AI prompts
            </div>
            <div>
              <span>{userAnswerCount}</span>
              Answers
            </div>
            <div>
              <span>{progress}%</span>
              Progress
            </div>
          </section>

          <section className="focus-panel">
            <div className="panel-heading">
              <Activity size={16} />
              Evaluation focus
            </div>
            <div className="focus-list">
              {focusAreas.map((area) => (
                <span key={area}>{area}</span>
              ))}
            </div>
          </section>

          <section className="transcript-panel">
            <div className="panel-heading transcript-heading">
              Transcript
              <span>{messages.length + (isPendingSubmit ? 1 : 0)}</span>
            </div>

            <div className="transcript-body">
              {messages.length === 0 && !isPendingSubmit ? (
                <div className="empty-state">
                  <BrainCircuit size={34} />
                  <p>The interview transcript and editable answer review will appear here.</p>
                </div>
              ) : (
                <>
                  {messages.map((message, index) => (
                    <article key={`${message.role}-${index}`} className={`message-card ${message.role}`}>
                      <p>{message.role === "ai" ? "InterVo" : "Candidate"}</p>
                      <div>{message.text}</div>
                    </article>
                  ))}

                  {isPendingSubmit && (
                    <article className="draft-card">
                      <label htmlFor="answer-draft">Review transcribed answer</label>
                      <textarea
                        id="answer-draft"
                        value={draftAnswer}
                        onChange={(event) => setDraftAnswer(event.target.value)}
                        placeholder="Your answer transcript will appear here."
                        rows={5}
                      />
                      <div className="draft-actions">
                        <button className="secondary-action" onClick={discardDraft}>
                          <RefreshCw size={15} />
                          Re-record
                        </button>
                        <button className="submit-action" onClick={submitAnswer} disabled={!draftAnswer.trim()}>
                          <Send size={15} />
                          Submit
                        </button>
                      </div>
                    </article>
                  )}
                  <div ref={messagesEndRef} />
                </>
              )}
            </div>
          </section>
        </aside>
      </main>
    </div>
  );
}
