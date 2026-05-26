"use client";

import React, { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  ClipboardList,
  Clock3,
  FileSearch,
  RefreshCw,
  ShieldAlert,
  Sparkles,
} from "lucide-react";

const SERVER_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ScoreEvidence {
  dimension: string;
  summary: string;
  transcript_excerpt: string | null;
  confidence: number;
}

interface DimensionScore {
  dimension: string;
  score: number | null;
  evidence: ScoreEvidence[];
}

interface QuestionTimelineEntry {
  question_id: string;
  question_title: string;
  source_type: string;
  state: string;
  prompt_text: string;
  answer_text: string;
  evaluation_summary: string;
  recommended_next_step: string;
  score_average: number | null;
  dimension_scores: DimensionScore[];
  evidence: ScoreEvidence[];
  concepts_demonstrated: string[];
  missing_concepts: string[];
  started_at: string | null;
  submitted_at: string | null;
  expired: boolean;
  lock_reason: string;
}

interface FinalScorecard {
  grading_complete: boolean;
  recommendation_ready: boolean;
  attempts_graded: number;
  questions_expected: number;
  recommendation: string | null;
  summary: string;
  overall_average: number | null;
  recommendation_rationale: string;
  recommendation_blocked_reason: string;
  dimension_scores: DimensionScore[];
  strengths: string[];
  risks: string[];
  unanswered_concerns: string[];
  evidence: ScoreEvidence[];
  question_summaries: QuestionTimelineEntry[];
  generated_at: string | null;
}

interface SessionEvent {
  at: string;
  kind: string;
  detail: string;
}

interface ScorecardPayload {
  session_id: string;
  status: string;
  template_id: string | null;
  scorecard: FinalScorecard;
  question_timeline: QuestionTimelineEntry[];
  session_events: SessionEvent[];
}

function getServerUrl() {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  if (typeof window === "undefined") return SERVER_URL;

  const { hostname } = window.location;
  if (hostname === "localhost" || hostname === "127.0.0.1") return SERVER_URL;

  return `http://${hostname}:8000`;
}

function humanize(value: string | null | undefined) {
  if (!value) return "Not available";
  return value.replace(/_/g, " ");
}

function formatScore(value: number | null | undefined) {
  return value === null || value === undefined ? "--" : value.toFixed(2);
}

function formatDate(value: string | null | undefined) {
  if (!value) return "Not recorded";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function recommendationClass(recommendation: string | null) {
  if (!recommendation) return "recommendation-muted";
  if (recommendation === "strong_hire" || recommendation === "hire") return "recommendation-positive";
  if (recommendation === "mixed") return "recommendation-caution";
  return "recommendation-negative";
}

export default function RecruiterReviewClient() {
  const [sessionId, setSessionId] = useState("");
  const [loadedSessionId, setLoadedSessionId] = useState("");
  const [payload, setPayload] = useState<ScorecardPayload | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const scorecard = payload?.scorecard;
  const timeline = payload?.question_timeline ?? [];

  const orderedEvents = useMemo(
    () => [...(payload?.session_events ?? [])].sort((a, b) => new Date(a.at).getTime() - new Date(b.at).getTime()),
    [payload?.session_events]
  );

  const loadScorecard = useCallback(async (id = sessionId) => {
    const trimmedId = id.trim();
    if (!trimmedId) {
      setError("Enter a completed interview session id.");
      return;
    }

    setError("");
    setLoading(true);

    try {
      const response = await fetch(`${getServerUrl()}/sessions/${trimmedId}/scorecard`);
      if (!response.ok) throw new Error(`Server returned ${response.status}`);

      const data = (await response.json()) as ScorecardPayload;
      setPayload(data);
      setLoadedSessionId(trimmedId);
    } catch (err) {
      console.error(err);
      setPayload(null);
      setError("Could not load that scorecard. Check the session id and backend status.");
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const querySessionId = params.get("session");
    if (querySessionId) {
      setSessionId(querySessionId);
      void loadScorecard(querySessionId);
    }
  }, [loadScorecard]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void loadScorecard();
  }

  return (
    <div className="recruiter-shell">
      <header className="recruiter-header">
        <div className="brand-block">
          <div className="brand-mark recruiter-mark">
            <ClipboardList size={21} />
          </div>
          <div>
            <p className="eyebrow">Recruiter review</p>
            <h1>InterVo Scorecard</h1>
          </div>
        </div>

        <form className="scorecard-search" onSubmit={handleSubmit}>
          <FileSearch size={17} />
          <input
            value={sessionId}
            onChange={(event) => setSessionId(event.target.value)}
            placeholder="Session id"
            aria-label="Session id"
          />
          <button type="submit" disabled={loading}>
            {loading ? <RefreshCw className="spin" size={16} /> : "Load"}
          </button>
        </form>
      </header>

      {error && (
        <div className="error-banner">
          <AlertCircle size={17} />
          {error}
        </div>
      )}

      {!payload ? (
        <main className="recruiter-empty">
          <FileSearch size={42} />
          <h2>Load a completed session scorecard</h2>
          <p>Use a session id from a candidate interview to review recommendation, evidence, answers, and audit events.</p>
        </main>
      ) : (
        <main className="recruiter-workspace">
          <section className="scorecard-summary-panel">
            <div className="summary-topline">
              <div>
                <p className="eyebrow">Session</p>
                <h2>{loadedSessionId}</h2>
                <span>{humanize(payload.status)} · {payload.template_id || "default template"}</span>
              </div>
              <div className={`recommendation-pill ${recommendationClass(scorecard?.recommendation || null)}`}>
                {humanize(scorecard?.recommendation)}
              </div>
            </div>

            <div className="summary-metrics">
              <div>
                <span>{formatScore(scorecard?.overall_average)}</span>
                Overall
              </div>
              <div>
                <span>{scorecard?.attempts_graded ?? 0}/{scorecard?.questions_expected ?? 0}</span>
                Graded
              </div>
              <div>
                <span>{scorecard?.recommendation_ready ? "Ready" : "Blocked"}</span>
                Recommendation
              </div>
              <div>
                <span>{scorecard?.grading_complete ? "Complete" : "Partial"}</span>
                Rubrics
              </div>
            </div>

            <p className="scorecard-summary-text">
              {scorecard?.summary || scorecard?.recommendation_blocked_reason || "No scorecard summary is available yet."}
            </p>

            {scorecard?.recommendation_rationale && (
              <div className="rationale-box">
                <Sparkles size={17} />
                {scorecard.recommendation_rationale}
              </div>
            )}
          </section>

          <section className="review-grid">
            <article className="review-panel">
              <div className="panel-heading">
                <CheckCircle2 size={16} />
                Strengths
              </div>
              <div className="tag-list positive-tags">
                {(scorecard?.strengths.length ? scorecard.strengths : ["No strong signals recorded"]).map((item) => (
                  <span key={item}>{humanize(item)}</span>
                ))}
              </div>
            </article>

            <article className="review-panel">
              <div className="panel-heading">
                <ShieldAlert size={16} />
                Risks
              </div>
              <div className="tag-list risk-tags">
                {(scorecard?.risks.length ? scorecard.risks : ["No major risks recorded"]).map((item) => (
                  <span key={item}>{humanize(item)}</span>
                ))}
              </div>
            </article>

            <article className="review-panel review-panel-wide">
              <div className="panel-heading">
                <AlertCircle size={16} />
                Unanswered concerns
              </div>
              <div className="concern-list">
                {(scorecard?.unanswered_concerns.length
                  ? scorecard.unanswered_concerns
                  : ["No unanswered concerns recorded."]
                ).map((item) => (
                  <p key={item}>{item}</p>
                ))}
              </div>
            </article>
          </section>

          <section className="review-panel">
            <div className="panel-heading">
              <BarChart3 size={16} />
              Rubric scores
            </div>
            <div className="dimension-grid">
              {(scorecard?.dimension_scores ?? []).map((dimension) => (
                <article key={dimension.dimension} className="dimension-card">
                  <div>
                    <strong>{humanize(dimension.dimension)}</strong>
                    <span>{formatScore(dimension.score)} / 5</span>
                  </div>
                  <div className="score-bar">
                    <i style={{ width: `${Math.min(100, ((dimension.score ?? 0) / 5) * 100)}%` }} />
                  </div>
                  {dimension.evidence[0] && <p>{dimension.evidence[0].summary}</p>}
                </article>
              ))}
              {scorecard?.dimension_scores.length === 0 && <p className="muted-copy">No rubric scores are available.</p>}
            </div>
          </section>

          <section className="review-panel">
            <div className="panel-heading">
              <ClipboardList size={16} />
              Question timeline
            </div>
            <div className="timeline-list">
              {timeline.map((entry, index) => (
                <article key={`${entry.question_id}-${entry.source_type}-${index}`} className="timeline-card">
                  <div className="timeline-card-header">
                    <div>
                      <p className="eyebrow">{entry.source_type === "follow_up" ? "Follow-up" : `Question ${index + 1}`}</p>
                      <h3>{entry.question_title}</h3>
                    </div>
                    <span className={`attempt-state attempt-state-${entry.state}`}>{humanize(entry.state)}</span>
                  </div>

                  <div className="timeline-copy">
                    <strong>Prompt</strong>
                    <p>{entry.prompt_text || "Prompt not recorded."}</p>
                  </div>

                  <div className="timeline-copy">
                    <strong>Answer</strong>
                    <p>{entry.answer_text || "No answer captured."}</p>
                  </div>

                  <div className="timeline-meta">
                    <span>Average: {formatScore(entry.score_average)}</span>
                    <span>Started: {formatDate(entry.started_at)}</span>
                    <span>Submitted: {formatDate(entry.submitted_at)}</span>
                    {entry.expired && <span>Expired: {humanize(entry.lock_reason)}</span>}
                  </div>

                  {entry.evaluation_summary && (
                    <div className="timeline-copy">
                      <strong>Evaluation</strong>
                      <p>{entry.evaluation_summary}</p>
                    </div>
                  )}

                  <div className="evidence-list">
                    {entry.evidence.map((evidence, evidenceIndex) => (
                      <blockquote key={`${evidence.dimension}-${evidenceIndex}`}>
                        <strong>{humanize(evidence.dimension)}</strong>
                        <span>{evidence.summary}</span>
                        {evidence.transcript_excerpt && <cite>{evidence.transcript_excerpt}</cite>}
                      </blockquote>
                    ))}
                  </div>
                </article>
              ))}
            </div>
          </section>

          <section className="review-panel">
            <div className="panel-heading">
              <Clock3 size={16} />
              Session events
            </div>
            <div className="event-list">
              {orderedEvents.map((event, index) => (
                <article key={`${event.kind}-${event.at}-${index}`}>
                  <span>{formatDate(event.at)}</span>
                  <strong>{humanize(event.kind)}</strong>
                  <p>{event.detail}</p>
                </article>
              ))}
              {orderedEvents.length === 0 && <p className="muted-copy">No audit events are available.</p>}
            </div>
          </section>
        </main>
      )}
    </div>
  );
}
