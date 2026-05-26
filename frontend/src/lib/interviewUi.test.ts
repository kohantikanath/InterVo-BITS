import { describe, expect, it } from "vitest";

import { buildTranscript, formatState, formatTime, isQuestionLocked, isTimerWarning } from "./interviewUi";

describe("interview UI helpers", () => {
  it("formats countdown values without negative display drift", () => {
    expect(formatTime(null)).toBe("--:--");
    expect(formatTime(0)).toBe("00:00");
    expect(formatTime(65)).toBe("01:05");
    expect(formatTime(-4)).toBe("00:00");
  });

  it("builds transcript review messages from attempts", () => {
    const messages = buildTranscript([
      {
        id: "attempt-1",
        state: "scored",
        prompt_text: "Explain two sum.",
        ai_text: "",
        user_text: "Use a hash map.",
      },
      {
        id: "attempt-2",
        state: "expired",
        prompt_text: "Explain graph traversal.",
        ai_text: "",
        user_text: "",
      },
    ]);

    expect(messages).toEqual([
      { role: "ai", text: "Explain two sum.", attemptId: "attempt-1" },
      { role: "user", text: "Use a hash map.", attemptId: "attempt-1" },
      { role: "ai", text: "Explain graph traversal.", attemptId: "attempt-2" },
      {
        role: "system",
        text: "Time expired before an answer was submitted.",
        attemptId: "attempt-2",
      },
    ]);
  });

  it("marks warning and locked timer states for candidate controls", () => {
    const timer = {
      warning_threshold_seconds: 30,
      expired: false,
      locked: false,
    };

    expect(isTimerWarning(timer, 31, false)).toBe(false);
    expect(isTimerWarning(timer, 30, false)).toBe(true);
    expect(isTimerWarning(timer, 0, false)).toBe(false);
    expect(isTimerWarning(timer, 20, true)).toBe(false);

    expect(isQuestionLocked("active", false)).toBe(false);
    expect(isQuestionLocked("expired", false)).toBe(true);
    expect(isQuestionLocked("scored", false)).toBe(true);
    expect(isQuestionLocked("active", true)).toBe(true);
  });

  it("humanizes backend state labels", () => {
    expect(formatState("review_pending")).toBe("review pending");
  });
});
