export type QuestionState = "ready" | "active" | "warning" | "expired" | "submitted" | "scored" | "advanced";

export interface TranscriptAttempt {
  id: string;
  state: QuestionState;
  prompt_text: string;
  ai_text: string;
  user_text: string;
}

export interface TranscriptMessage {
  role: "ai" | "user" | "system";
  text: string;
  attemptId?: string;
}

export interface TimerLike {
  warning_threshold_seconds: number;
  expired: boolean;
  locked: boolean;
}

export function formatTime(seconds: number | null | undefined) {
  if (seconds === null || seconds === undefined) return "--:--";
  const safeSeconds = Math.max(0, seconds);
  const mins = Math.floor(safeSeconds / 60)
    .toString()
    .padStart(2, "0");
  const secs = (safeSeconds % 60).toString().padStart(2, "0");
  return `${mins}:${secs}`;
}

export function formatState(value: string) {
  return value.replace(/_/g, " ");
}

export function buildTranscript(attempts: TranscriptAttempt[]): TranscriptMessage[] {
  const messages: TranscriptMessage[] = [];

  attempts.forEach((attempt) => {
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

export function isTimerWarning(
  timer: TimerLike | null,
  remainingSeconds: number | null,
  attemptLocked: boolean,
) {
  return Boolean(
    timer &&
      remainingSeconds !== null &&
      remainingSeconds > 0 &&
      remainingSeconds <= timer.warning_threshold_seconds &&
      !attemptLocked,
  );
}

export function isQuestionLocked(state: QuestionState | null | undefined, timerLocked = false) {
  return Boolean(timerLocked || state === "expired" || state === "scored" || state === "advanced");
}
