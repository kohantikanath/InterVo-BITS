"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  Bot,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  FileText,
  Lock,
  Mic,
  Play,
  RefreshCw,
  Send,
  ShieldCheck,
  Square,
  User,
} from "lucide-react";

const SERVER_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Phase =
  | "idle"
  | "starting"
  | "ready"
  | "recording"
  | "transcribing"
  | "pending_submit"
  | "submitting"
  | "advancing"
  | "completed";

type QuestionState = "ready" | "active" | "warning" | "expired" | "submitted" | "scored" | "advanced";
type SessionStatus = "draft" | "ready" | "in_progress" | "review_pending" | "completed" | "aborted";

interface Message {
  role: "ai" | "user" | "system";
  text: string;
  attemptId?: string;
}

interface QuestionAttempt {
  id: string;
  question_id: string | null;
  question_title: string | null;
  state: QuestionState;
  source_type: string;
  prompt_text: string;
  user_text: string;
  ai_text: string;
  evaluation_summary: string;
  recommended_next_step: string;
  started_at: string | null;
  submitted_at: string | null;
  time_cap_seconds: number | null;
  warning_threshold_seconds: number;
  expires_at: string | null;
  lock_reason: string;
}

interface InterviewSession {
  id: string;
  status: SessionStatus;
  mode: "hiring" | "practice";
  attempts: QuestionAttempt[];
  current_attempt_id: string | null;
  last_ai_text: string;
  started_at: string | null;
  completed_at: string | null;
}

interface TimerSnapshot {
  state: QuestionState;
  expires_at: string | null;
  warning_threshold_seconds: number;
  remaining_seconds: number | null;
  expired: boolean;
  locked: boolean;
  lock_reason: string;
}

interface SessionPayload {
  session: InterviewSession;
  current_attempt: QuestionAttempt | null;
  timer: TimerSnapshot | null;
  ai_text?: string;
  status?: string;
  error?: string;
}

const phaseLabels: Record<Phase, string> = {
  idle: "Ready",
  starting: "Starting",
  ready: "Candidate turn",
  recording: "Recording",
  transcribing: "Transcribing",
  pending_submit: "Review transcript",
  submitting: "Evaluating",
  advancing: "Advancing",
  completed: "Completed",
};

const lockedQuestionStates: QuestionState[] = ["expired", "scored", "advanced"];

function formatTime(seconds: number | null | undefined) {
  if (seconds === null || seconds === undefined) return "--:--";
  const safeSeconds = Math.max(0, seconds);
  const mins = Math.floor(safeSeconds / 60)
    .toString()
    .padStart(2, "0");
  const secs = (safeSeconds % 60).toString().padStart(2, "0");
  return `${mins}:${secs}`;
}

function formatState(value: string) {
  return value.replace(/_/g, " ");
}

function getServerUrl() {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  if (typeof window === "undefined") return SERVER_URL;

  const { hostname } = window.location;
  if (hostname === "localhost" || hostname === "127.0.0.1") return SERVER_URL;

  return `http://${hostname}:8000`;
}

function buildTranscript(session: InterviewSession): Message[] {
  const messages: Message[] = [];

  session.attempts.forEach((attempt) => {
    const prompt = attempt.prompt_text || attempt.ai_text;
    if (prompt) {
      messages.push({
        role: "ai",
        text: prompt,
        attemptId: attempt.id,
      });
    }

    if (attempt.user_text) {
      messages.push({
        role: "user",
        text: attempt.user_text,
        attemptId: attempt.id,
      });
    }

    if (attempt.state === "expired" && !attempt.user_text) {
      messages.push({
        role: "system",
        text: "Time expired before an answer was submitted.",
        attemptId: attempt.id,
      });
    }
  });

  return messages;
}

export default function InterviewClient() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [session, setSession] = useState<InterviewSession | null>(null);
  const [currentAttempt, setCurrentAttempt] = useState<QuestionAttempt | null>(null);
  const [timer, setTimer] = useState<TimerSnapshot | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [draftAnswer, setDraftAnswer] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [localRemainingSeconds, setLocalRemainingSeconds] = useState<number | null>(null);
  const [warningSentForAttemptId, setWarningSentForAttemptId] = useState<string | null>(null);
  const [expirySentForAttemptId, setExpirySentForAttemptId] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const isRecording = phase === "recording";
  const isPendingSubmit = phase === "pending_submit";
  const isBusy = ["starting", "submitting", "transcribing", "advancing"].includes(phase);
  const sessionStarted = Boolean(session);
  const isCompleted = session?.status === "completed" || phase === "completed";
  const attemptLocked = Boolean(timer?.locked || (currentAttempt && lockedQuestionStates.includes(currentAttempt.state)));
  const isExpired = Boolean(timer?.expired || currentAttempt?.state === "expired");
  const showWarning = Boolean(
    timer &&
      localRemainingSeconds !== null &&
      localRemainingSeconds > 0 &&
      localRemainingSeconds <= timer.warning_threshold_seconds &&
      !attemptLocked
  );

  const primaryAttempts = session?.attempts.filter((attempt) => attempt.source_type === "question") ?? [];
  const answeredAttempts = primaryAttempts.filter((attempt) =>
    ["scored", "advanced", "expired"].includes(attempt.state)
  );
  const expectedQuestions = 6;
  const progress = Math.min(100, Math.round((answeredAttempts.length / expectedQuestions) * 100));
  const statusLabel = isExpired ? "Expired" : showWarning ? "Time warning" : phaseLabels[phase];

  const currentGuidance = useMemo(() => {
    if (!sessionStarted) return "Begin a hiring-grade DSA interview session.";
    if (isCompleted) return "Interview complete. The recruiter scorecard can be reviewed separately.";
    if (isExpired) return "The answer window is locked. Advance to continue.";
    if (showWarning) return "Wrap up your reasoning. The hard time cap is approaching.";
    if (phase === "recording") return "Answer with assumptions, algorithm, complexity, and edge cases.";
    if (phase === "pending_submit") return "Review the transcript. No hints or correctness checks are shown in hiring mode.";
    if (phase === "transcribing") return "Converting your voice into editable transcript text.";
    if (phase === "submitting") return "Evaluating this attempt against the hiring rubric.";
    if (phase === "advancing") return "Loading the next interview step.";
    return "Record your answer when ready.";
  }, [isCompleted, isExpired, phase, sessionStarted, showWarning]);

  const applySessionPayload = useCallback((payload: SessionPayload, fallbackPhase?: Phase) => {
    setSession(payload.session);
    setCurrentAttempt(payload.current_attempt);
    setTimer(payload.timer);
    setMessages(buildTranscript(payload.session));
    setLocalRemainingSeconds(payload.timer?.remaining_seconds ?? null);

    if (payload.session.status === "completed") {
      setPhase("completed");
      setNotice("Interview completed. Recruiter scorecard generation is available from the backend.");
      return;
    }

    if (payload.timer?.expired || payload.current_attempt?.state === "expired") {
      setPhase("ready");
      setNotice("Time is up. The current answer window is locked.");
      return;
    }

    if (payload.session.status === "review_pending" || payload.current_attempt?.state === "scored") {
      setPhase("ready");
      setNotice("Answer recorded and evaluated. Advance when ready.");
      return;
    }

    if (fallbackPhase) {
      setPhase(fallbackPhase);
      return;
    }

    setPhase("ready");
  }, []);

  function setFailure(message: string) {
    setError(message);
    setPhase(session ? "ready" : "idle");
  }

  async function requestJson(path: string, init?: RequestInit): Promise<SessionPayload> {
    const response = await fetch(`${getServerUrl()}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...init?.headers,
      },
    });

    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    return response.json();
  }

  async function beginInterview() {
    setError("");
    setNotice("");
    setPhase("starting");
    setDraftAnswer("");

    try {
      const created = await requestJson("/sessions", {
        method: "POST",
        body: JSON.stringify({ mode: "hiring" }),
      });
      const started = await requestJson(`/sessions/${created.session.id}/start`, { method: "POST" });
      applySessionPayload(started, "ready");
    } catch (err) {
      console.error(err);
      setFailure("Backend is not reachable. Start FastAPI on port 8000 and try again.");
    }
  }

  async function refreshSession() {
    if (!session?.id) return;
    setError("");

    try {
      const payload = await requestJson(`/sessions/${session.id}`);
      applySessionPayload(payload);
    } catch (err) {
      console.error(err);
      setFailure("Could not refresh the interview session.");
    }
  }

  const recordTimerEvent = useCallback(async (event: "sync" | "warning" | "expire") => {
    if (!session?.id) return;

    try {
      const payload = await requestJson(`/sessions/${session.id}/timer-events`, {
        method: "POST",
        body: JSON.stringify({ event }),
      });
      applySessionPayload(payload);
    } catch (err) {
      console.error(err);
      setError("Timer sync failed. Refresh before continuing.");
    }
  }, [applySessionPayload, session?.id]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, draftAnswer, phase]);

  useEffect(() => {
    return () => {
      mediaRecorderRef.current?.stream.getTracks().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    if (!timer?.expires_at || attemptLocked || isCompleted) {
      setLocalRemainingSeconds(timer?.remaining_seconds ?? null);
      return;
    }

    const updateRemaining = () => {
      const expiresAt = new Date(timer.expires_at as string).getTime();
      setLocalRemainingSeconds(Math.max(0, Math.ceil((expiresAt - Date.now()) / 1000)));
    };

    updateRemaining();
    const interval = window.setInterval(updateRemaining, 1000);
    return () => window.clearInterval(interval);
  }, [attemptLocked, isCompleted, timer]);

  useEffect(() => {
    if (!session?.id || !currentAttempt || !timer || attemptLocked) return;

    if (
      localRemainingSeconds !== null &&
      localRemainingSeconds > 0 &&
      localRemainingSeconds <= timer.warning_threshold_seconds &&
      warningSentForAttemptId !== currentAttempt.id
    ) {
      setWarningSentForAttemptId(currentAttempt.id);
      void recordTimerEvent("warning");
    }

    if (
      localRemainingSeconds === 0 &&
      expirySentForAttemptId !== currentAttempt.id &&
      currentAttempt.state !== "expired"
    ) {
      setExpirySentForAttemptId(currentAttempt.id);
      void recordTimerEvent("expire");
    }
  }, [
    attemptLocked,
    currentAttempt,
    expirySentForAttemptId,
    localRemainingSeconds,
    recordTimerEvent,
    session?.id,
    timer,
    warningSentForAttemptId,
  ]);

  async function startRecording() {
    if (attemptLocked || isCompleted) return;
    setError("");
    setNotice("");

    try {
      if (!window.isSecureContext) {
        throw new Error(
          "Microphone access only works on localhost or HTTPS. Open the app with http://localhost:3000 on this computer."
        );
      }

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("This browser does not expose microphone recording on the current page.");
      }

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
      setFailure(err instanceof Error ? err.message : "Microphone permission is required for the interview.");
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
    if (attemptLocked) {
      setPhase("ready");
      return;
    }

    try {
      const formData = new FormData();
      const ext = blob.type.includes("ogg") ? "ogg" : blob.type.includes("mp4") ? "mp4" : "webm";
      formData.append("file", blob, `recording.${ext}`);

      const response = await fetch(`${getServerUrl()}/transcribe-audio`, {
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
    if (!session?.id || !text || attemptLocked) return;

    setError("");
    setNotice("");
    setPhase("submitting");

    try {
      const payload = await requestJson(`/sessions/${session.id}/answers`, {
        method: "POST",
        body: JSON.stringify({ user_text: text }),
      });

      if (payload.error) {
        setError(payload.error);
      }

      setDraftAnswer("");
      applySessionPayload(payload);
    } catch (err) {
      console.error(err);
      setFailure("Could not submit the answer. Confirm the backend and model settings are configured.");
    }
  }

  async function advanceInterview() {
    if (!session?.id || isBusy) return;

    setError("");
    setNotice("");
    setPhase("advancing");

    try {
      const payload = await requestJson(`/sessions/${session.id}/advance`, {
        method: "POST",
        body: JSON.stringify({ reason: isExpired ? "candidate_ui_expired_advance" : "candidate_ui_manual_advance" }),
      });

      if (payload.error) {
        setError(payload.error);
      }

      setDraftAnswer("");
      setWarningSentForAttemptId(null);
      setExpirySentForAttemptId(null);
      applySessionPayload(payload);
    } catch (err) {
      console.error(err);
      setFailure("Could not advance the interview session.");
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
            <p className="eyebrow">Hiring-grade DSA interview</p>
            <h1>InterVo</h1>
          </div>
        </div>

        <div className={`live-status live-status-${isExpired ? "expired" : showWarning ? "warning" : phase}`}>
          <span />
          {statusLabel}
        </div>

        <div className={`session-time ${showWarning ? "session-time-warning" : ""} ${isExpired ? "session-time-expired" : ""}`}>
          {attemptLocked ? <Lock size={16} /> : <Clock3 size={16} />}
          {formatTime(localRemainingSeconds)}
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <AlertCircle size={17} />
          {error}
        </div>
      )}

      {notice && !error && (
        <div className="notice-banner">
          <CheckCircle2 size={17} />
          {notice}
        </div>
      )}

      <main className="workspace">
        <section className={`stage-panel ${isExpired ? "stage-expired" : ""}`}>
          <div className="stage-header">
            <div>
              <p className="eyebrow">Candidate interview room</p>
              <h2>{currentAttempt?.question_title || "Structured DSA assessment"}</h2>
            </div>
            <div className="question-meter">
              <span>{answeredAttempts.length}/{expectedQuestions}</span>
              primary questions
            </div>
          </div>

          <div className="progress-track" aria-label="Interview progress">
            <div style={{ width: `${progress}%` }} />
          </div>

          <section className="question-panel">
            <div className="question-panel-heading">
              <div>
                <p className="eyebrow">{currentAttempt?.source_type === "follow_up" ? "Follow-up" : "Current question"}</p>
                <h3>{currentAttempt?.prompt_text || "Start the interview to load the first question."}</h3>
              </div>
              <span className={`attempt-state attempt-state-${currentAttempt?.state || "ready"}`}>
                {currentAttempt ? formatState(currentAttempt.state) : "not started"}
              </span>
            </div>
            {currentAttempt?.time_cap_seconds && (
              <div className="time-cap-row">
                <Clock3 size={16} />
                Hard cap: {formatTime(currentAttempt.time_cap_seconds)}
              </div>
            )}
          </section>

          <div className="participant-grid">
            <article className="person-card interviewer">
              <div className="person-background" />
              <div className="avatar-ring">
                <Bot size={52} />
              </div>
              <div className="person-footer">
                <div>
                  <strong>InterVo</strong>
                  <span>Strict hiring evaluator</span>
                </div>
                <ShieldCheck size={16} />
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
                  <span>{attemptLocked ? "Answer locked" : isRecording ? "Answering now" : "Waiting turn"}</span>
                </div>
                {attemptLocked ? <Lock size={16} /> : <Mic size={16} />}
              </div>
            </article>
          </div>

          <div className="control-strip">
            {!sessionStarted ? (
              <button className="primary-action" onClick={beginInterview} disabled={phase === "starting"}>
                {phase === "starting" ? <RefreshCw className="spin" size={18} /> : <Play size={18} />}
                {phase === "starting" ? "Starting" : "Begin interview"}
              </button>
            ) : isCompleted ? (
              <button className="secondary-action" onClick={refreshSession}>
                <RefreshCw size={16} />
                Refresh
              </button>
            ) : isPendingSubmit ? (
              <div className="review-reminder">
                <FileText size={18} />
                Transcript review required before submission.
              </div>
            ) : attemptLocked || session?.status === "review_pending" ? (
              <button className="primary-action" onClick={advanceInterview} disabled={isBusy}>
                {phase === "advancing" ? <RefreshCw className="spin" size={18} /> : <Send size={18} />}
                Advance
              </button>
            ) : (
              <button
                className={`record-action ${isRecording ? "recording" : ""}`}
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isBusy}
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
              <span>{primaryAttempts.length}</span>
              Loaded
            </div>
            <div>
              <span>{answeredAttempts.length}</span>
              Resolved
            </div>
            <div>
              <span>{progress}%</span>
              Progress
            </div>
          </section>

          <section className="focus-panel">
            <div className="panel-heading">
              <ShieldCheck size={16} />
              Hiring mode
            </div>
            <div className="policy-list">
              <span>No hints</span>
              <span>No solution coaching</span>
              <span>No grading disclosure</span>
              <span>Timer enforced</span>
            </div>
          </section>

          <section className="transcript-panel">
            <div className="panel-heading transcript-heading">
              Transcript review
              <span>{messages.length + (isPendingSubmit ? 1 : 0)}</span>
            </div>

            <div className="transcript-body">
              {messages.length === 0 && !isPendingSubmit ? (
                <div className="empty-state">
                  <BrainCircuit size={34} />
                  <p>The session transcript and answer review will appear here.</p>
                </div>
              ) : (
                <>
                  {messages.map((message, index) => (
                    <article key={`${message.role}-${message.attemptId || index}-${index}`} className={`message-card ${message.role}`}>
                      <p>{message.role === "ai" ? "InterVo" : message.role === "user" ? "Candidate" : "System"}</p>
                      <div>{message.text}</div>
                    </article>
                  ))}

                  {isPendingSubmit && (
                    <article className={`draft-card ${attemptLocked ? "draft-locked" : ""}`}>
                      <label htmlFor="answer-draft">Review transcribed answer</label>
                      <textarea
                        id="answer-draft"
                        value={draftAnswer}
                        onChange={(event) => setDraftAnswer(event.target.value)}
                        placeholder="Your answer transcript will appear here."
                        rows={5}
                        disabled={attemptLocked}
                      />
                      <div className="draft-actions">
                        <button className="secondary-action" onClick={discardDraft} disabled={attemptLocked}>
                          <RefreshCw size={15} />
                          Re-record
                        </button>
                        <button className="submit-action" onClick={submitAnswer} disabled={!draftAnswer.trim() || attemptLocked}>
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
