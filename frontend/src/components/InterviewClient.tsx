"use client";

import React, { useState, useRef, useEffect } from "react";
import { Mic, Square, Play, Send, RefreshCw, Bot, User, BrainCircuit } from "lucide-react";

// ─── Backend URL ─────────────────────────────────────────────────────────────
const SERVER_URL = "http://localhost:8000";

type Phase = "idle" | "ai_speaking" | "ready" | "recording" | "transcribing" | "pending_submit" | "submitting";

interface Message {
  role: "ai" | "user";
  text: string;
  pending?: boolean;
}

export default function InterviewClient() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [interviewStarted, setInterviewStarted] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [draftAnswer, setDraftAnswer] = useState("");

  const webMediaRecorderRef = useRef<MediaRecorder | null>(null);
  const webAudioChunksRef = useRef<Blob[]>([]);
  const tsBodyRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const scrollToBottom = () => {
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }, 100);
  };

  const addMessage = (role: "ai" | "user", text: string, pending = false) => {
    setMessages((prev) => [...prev, { role, text, pending }]);
    scrollToBottom();
  };

  useEffect(() => {
    return () => {
      // Cleanup audio on unmount
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = "";
      }
    };
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, phase, draftAnswer]);

  // ── Status label derived from phase ────────────────────────────────────────
  const statusMap: Record<Phase, string> = {
    idle: "Ready",
    ai_speaking: "InterVo Speaking",
    ready: "Your Turn",
    recording: "Recording",
    transcribing: "Transcribing…",
    pending_submit: "Review Answer",
    submitting: "Processing…",
  };

  const getPillClass = () => {
    if (phase === "ai_speaking") return "speaking";
    if (phase === "recording") return "recording";
    if (["transcribing", "submitting"].includes(phase)) return "thinking";
    return "ready-status";
  };
  const statusLabel = statusMap[phase] || "Ready";

  // ── 1. Begin Interview ───────────────────────────────────────────────────────
  async function beginInterview() {
    setPhase("submitting");
    try {
      const res = await fetch(`${SERVER_URL}/start-interview`, { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        setInterviewStarted(true);
        await playAiResponse(data.ai_text || null);
      } else {
        setPhase("idle");
        alert("Server error starting interview.");
      }
    } catch (e) {
      console.error(e);
      setPhase("idle");
    }
  }

  // ── 2. Play AI audio + show AI transcript immediately when done ──────────────
  async function playAiResponse(aiTextHint: string | null) {
    setPhase("ai_speaking");
    try {
      let aiText = aiTextHint;
      if (!aiText) {
        try {
          const r = await fetch(`${SERVER_URL}/get-ai-transcript`);
          if (r.ok) {
            const d = await r.json();
            aiText = d.ai_text;
          }
        } catch (_) {}
      }

      if (audioRef.current) {
        audioRef.current.pause();
      }

      const audio = new Audio(`${SERVER_URL}/get-audio?rnd=${Math.random()}`);
      audioRef.current = audio;

      audio.onended = () => {
        addMessage("ai", aiText || "AI responded.");
        setPhase("ready");
      };

      await audio.play();
    } catch (err) {
      console.error("Playback error:", err);
      // Fallback
      addMessage("ai", aiTextHint || "InterVo responded.");
      setPhase("ready");
    }
  }

  // ── 3. Start Recording ───────────────────────────────────────────────────────
  async function startRecording() {
    try {
      if (audioRef.current) {
        audioRef.current.pause();
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      const mr = new MediaRecorder(stream, { mimeType });
      
      webAudioChunksRef.current = [];
      mr.ondataavailable = (e) => {
        if (e.data.size > 0) webAudioChunksRef.current.push(e.data);
      };
      
      webMediaRecorderRef.current = mr;
      mr.start(100);
      setPhase("recording");
    } catch (err) {
      console.error(err);
      alert("Microphone access denied or not available.");
    }
  }

  // ── 4. Stop Recording → Transcribe only ─────────────────────────────────────
  async function stopRecording() {
    if (phase !== "recording") return;
    setPhase("transcribing");

    const mr = webMediaRecorderRef.current;
    if (!mr) return;

    mr.onstop = async () => {
      const blob = new Blob(webAudioChunksRef.current, { type: mr.mimeType });
      mr.stream.getTracks().forEach((t) => t.stop()); // Stop mic tracks
      await transcribeAudio(blob);
    };

    mr.stop();
  }

  // ── 5. Transcribe audio → show editable draft ───────────────────────────────
  async function transcribeAudio(blob: Blob) {
    try {
      const formData = new FormData();
      const ext = blob.type.includes("ogg") ? "ogg" : blob.type.includes("mp4") ? "mp4" : "webm";
      formData.append("file", blob, `recording.${ext}`);

      const res = await fetch(`${SERVER_URL}/transcribe-audio`, {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        const text = data.user_text || "";
        setDraftAnswer(text);
        setPhase("pending_submit");
        scrollToBottom();
      } else {
        console.error("Transcription failed", res.status);
        setPhase("ready");
      }
    } catch (err) {
      console.error("Transcribe error:", err);
      setPhase("ready");
    }
  }

  // ── 6. Submit (possibly edited) answer ──────────────────────────────────────
  async function submitAnswer() {
    const text = draftAnswer.trim();
    setPhase("submitting");

    addMessage("user", text || "(no answer)");
    setDraftAnswer("");

    try {
      const res = await fetch(`${SERVER_URL}/submit-answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_text: text }),
      });
      if (res.ok) {
        const data = await res.json();
        await playAiResponse(data.ai_text || null);
      } else {
        console.error("Submit failed", res.status);
        setPhase("ready");
      }
    } catch (err) {
      console.error("Submit error:", err);
      setPhase("ready");
    }
  }

  // ── 7. Re-record — discard draft ────────────────────────────────────────────
  function discardDraft() {
    setDraftAnswer("");
    setPhase("ready");
  }

  const isAiSpeaking = phase === "ai_speaking";
  const isRecording = phase === "recording";
  const isBusy = ["submitting", "transcribing", "ai_speaking"].includes(phase);
  const isPendingSubmit = phase === "pending_submit";
  const totalMsgs = messages.length + (isPendingSubmit ? 1 : 0);

  return (
    <div className="meet-shell">
      {/* ── Top bar ── */}
      <header className="meet-topbar">
        <div className="topbar-brand">
          <div className="topbar-logo"><BrainCircuit size={18} color="white" /></div>
          <div>
            <span className="topbar-title">InterVo</span>
            <span className="topbar-sub">Mathematics Admission</span>
          </div>
        </div>
        
        <div className={`status-pill ${getPillClass()}`}>
          <div className="status-dot" />
          {statusLabel}
        </div>
        
        <div className="timer-label">
          {interviewStarted
            ? phase === "ready"
              ? "Your turn to answer"
              : statusLabel
            : "Not started"}
        </div>
      </header>

      {/* ── Body ── */}
      <div className="meet-body">
        {/* ── Participants (left) ── */}
        <div className="participants-area">
          {/* AI tile */}
          <div className={`participant-tile ai-tile glass-panel ${isAiSpeaking ? "active ai-speaking-glow" : ""}`}>
            <div className="tile-bg" />
            <div className="tile-avatar-wrap">
              <div className="tile-avatar">
                {isBusy && !isAiSpeaking ? (
                  <div className="spinner" />
                ) : (
                  <Bot size={44} color="#a5b4fc" />
                )}
              </div>
              <div className={`wave-bars ${isAiSpeaking ? "active" : ""}`}>
                {[1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="wave-bar" />
                ))}
              </div>
            </div>
            <div className="tile-label">
              <div className="tile-name">InterVo (Interviewer)</div>
              <div className="tile-mic-indicator">
                {isAiSpeaking ? <Mic size={14} color="#34d399" /> : <Mic size={14} color="#ef4444" />}
              </div>
            </div>
          </div>

          {/* User tile */}
          <div className={`participant-tile user-tile glass-panel ${isRecording ? "active" : ""}`}>
            <div className="tile-bg" />
            {isRecording && (
              <div className="pulse-rings">
                <div className="pulse-ring" />
                <div className="pulse-ring" />
                <div className="pulse-ring" />
              </div>
            )}
            <div className="tile-avatar-wrap">
              <div className="tile-avatar">
                <User size={44} color="#cbd5e1" />
              </div>
            </div>
            <div className="tile-label">
              <div className="tile-name">You</div>
              <div className="tile-mic-indicator">
                {isRecording ? <Mic size={14} color="#34d399" /> : <Mic size={14} color="#ef4444" />}
              </div>
            </div>
          </div>
        </div>

        {/* ── Transcript sidebar ── */}
        <aside className="transcript-sidebar glass-panel">
          <div className="ts-header">
            <h3>Transcript</h3>
            {totalMsgs > 0 && <span className="ts-count">{totalMsgs}</span>}
          </div>

          <div className="ts-body" ref={tsBodyRef}>
            {messages.length === 0 && !isPendingSubmit ? (
              <div className="ts-empty">
                <div className="ts-empty-icon"><RefreshCw size={32} /></div>
                <div>Transcript appears here as the interview progresses.</div>
              </div>
            ) : (
              <>
                {messages.map((m, i) => (
                  <div key={i} className={`msg ${m.role} glass-panel slide-in`}>
                    <div className="msg-role">
                      {m.role === "ai" ? "🤖 InterVo" : "🎙 You"}
                    </div>
                    <div className="msg-text">{m.text}</div>
                  </div>
                ))}

                {/* Editable draft — shown after transcription, before submit */}
                {isPendingSubmit && (
                  <div className="pending-answer-wrap slide-in">
                    <div className="pending-answer-label">🎙 Your Answer — Review & Edit</div>
                    <textarea
                      className="pending-answer-textarea"
                      value={draftAnswer}
                      onChange={(e) => setDraftAnswer(e.target.value)}
                      placeholder="Your transcribed answer will appear here…"
                      rows={4}
                    />
                    <div className="pending-answer-actions">
                      <button className="button-secondary cancel-btn" onClick={discardDraft} title="Re-record">
                        <RefreshCw size={14} /> Re-record
                      </button>
                      <button
                        className="button-success submit-btn"
                        onClick={submitAnswer}
                        disabled={!draftAnswer.trim()}
                      >
                        <Send size={14} /> Submit Answer
                      </button>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} style={{ height: 1, flexShrink: 0 }} />
              </>
            )}
          </div>
        </aside>
      </div>

      {/* ── Bottom controls ── */}
      <div className="meet-controls glass-panel">
        {!interviewStarted ? (
          <button
            className="button-primary begin-btn"
            onClick={beginInterview}
            disabled={phase === "submitting"}
          >
            {phase === "submitting" ? (
              <><RefreshCw size={18} className="spinner-icon" /> Connecting…</>
            ) : (
              <><Play size={18} /> Begin Interview</>
            )}
          </button>
        ) : isPendingSubmit ? (
          <span className="ctrl-hint highlight">
            Review your transcript and click Submit to answer InterVo.
          </span>
        ) : (
          <button
            className={`button-base ctrl-btn ${isRecording ? "recording-pulse" : "button-secondary"}`}
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isBusy || isPendingSubmit}
            title={isRecording ? "Stop & Transcribe" : "Start Recording"}
            style={{ borderRadius: '50%', width: '56px', height: '56px' }}
          >
            {isRecording ? <Square size={22} color="white" fill="white" /> : <Mic size={22} />}
          </button>
        )}
        {interviewStarted && !isPendingSubmit && (
          <div className="ctrl-hint">
            {isAiSpeaking
              ? "InterVo is speaking…"
              : phase === "transcribing"
              ? "Transcribing your audio…"
              : phase === "submitting"
              ? "InterVo is thinking…"
              : isRecording
              ? "Tap the square to stop & transcribe"
              : "Tap the microphone to answer"}
          </div>
        )}
      </div>
    </div>
  );
}
