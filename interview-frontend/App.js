import React, { useState, useRef } from 'react';
import { Platform } from 'react-native';
import { Audio } from 'expo-av';

// ─── Backend URL ─────────────────────────────────────────────────────────────
const SERVER_URL = 'http://localhost:8000';

// ─── Styles ───────────────────────────────────────────────────────────────────
const WEB_STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  html, body, #root {
    width: 100%; height: 100%;
    font-family: 'Inter', sans-serif;
    background: #0a0d14;
    color: #e2e8f0;
    overflow: hidden;
  }

  /* ── Full-screen shell ── */
  .meet-shell {
    display: flex; flex-direction: column;
    width: 100vw; height: 100vh;
    background: #0a0d14; overflow: hidden;
  }

  /* ── Top bar ── */
  .meet-topbar {
    flex-shrink: 0; height: 56px;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 20px;
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(16px); z-index: 10;
  }
  .topbar-brand { display: flex; align-items: center; gap: 10px; }
  .topbar-logo {
    width: 32px; height: 32px; border-radius: 8px;
    background: linear-gradient(135deg, #6366f1, #10b981);
    display: flex; align-items: center; justify-content: center; font-size: 15px;
  }
  .topbar-title { font-size: 15px; font-weight: 700; color: #f1f5f9; }
  .topbar-sub   { font-size: 11px; color: #475569; margin-left: 8px; }
  .status-pill {
    display: flex; align-items: center; gap: 7px;
    padding: 5px 14px; border-radius: 999px;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.09);
    font-size: 12px; font-weight: 500; color: #94a3b8; transition: all 0.3s ease;
  }
  .status-pill.recording { border-color: rgba(239,68,68,0.4); color: #f87171; background: rgba(239,68,68,0.08); }
  .status-pill.speaking  { border-color: rgba(16,185,129,0.4); color: #34d399; background: rgba(16,185,129,0.08); }
  .status-pill.thinking  { border-color: rgba(99,102,241,0.4); color: #a5b4fc; background: rgba(99,102,241,0.08); }
  .status-dot {
    width: 6px; height: 6px; border-radius: 50%; background: currentColor;
    animation: pulse-dot 1.4s ease-in-out infinite;
  }
  @keyframes pulse-dot { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
  .timer-label { font-size: 12px; color: #475569; }

  /* ── Body ── */
  .meet-body { flex: 1; display: flex; overflow: hidden; }

  /* ── Participants (left) ── */
  .participants-area {
    flex: 1; display: flex; flex-direction: column;
    padding: 12px; gap: 10px; overflow: hidden;
  }

  /* ── Single participant tile ── */
  .participant-tile {
    flex: 1; position: relative; border-radius: 16px; overflow: hidden;
    background: #111622; border: 1px solid rgba(255,255,255,0.07);
    display: flex; align-items: center; justify-content: center;
    transition: border-color 0.4s ease, box-shadow 0.4s ease;
  }
  .participant-tile.ai-tile.active {
    border-color: rgba(16,185,129,0.5);
    box-shadow: 0 0 0 2px rgba(16,185,129,0.18), inset 0 0 60px rgba(16,185,129,0.04);
  }
  .participant-tile.user-tile.active {
    border-color: rgba(239,68,68,0.5);
    box-shadow: 0 0 0 2px rgba(239,68,68,0.18), inset 0 0 60px rgba(239,68,68,0.04);
  }
  .tile-bg {
    position: absolute; inset: 0;
    background-image:
      linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
    background-size: 40px 40px;
  }

  /* Avatar */
  .tile-avatar-wrap {
    position: relative; display: flex;
    align-items: center; justify-content: center; z-index: 2;
  }
  .tile-avatar {
    width: 100px; height: 100px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; font-size: 44px;
    border: 2px solid rgba(255,255,255,0.1); position: relative; z-index: 2;
    transition: border-color 0.4s, box-shadow 0.4s;
  }
  .ai-tile .tile-avatar   { background: linear-gradient(135deg, #1e1b4b, #0f1729); }
  .user-tile .tile-avatar { background: linear-gradient(135deg, #0c1a12, #0a0f18); }
  .ai-tile.active .tile-avatar {
    border-color: rgba(16,185,129,0.7);
    box-shadow: 0 0 40px rgba(16,185,129,0.35);
    animation: ai-glow 1.5s ease-in-out infinite;
  }
  .user-tile.active .tile-avatar {
    border-color: rgba(239,68,68,0.7);
    box-shadow: 0 0 40px rgba(239,68,68,0.35);
    animation: user-glow 1s ease-in-out infinite;
  }
  @keyframes ai-glow   { 0%,100%{box-shadow:0 0 20px rgba(16,185,129,0.3);} 50%{box-shadow:0 0 55px rgba(16,185,129,0.6);} }
  @keyframes user-glow { 0%,100%{box-shadow:0 0 20px rgba(239,68,68,0.3);} 50%{box-shadow:0 0 55px rgba(239,68,68,0.6);} }

  /* Wave bars */
  .wave-bars {
    position: absolute; bottom: -32px;
    display: flex; align-items: center; gap: 4px; height: 28px;
    opacity: 0; transition: opacity 0.3s;
  }
  .wave-bars.active { opacity: 1; }
  .wave-bar {
    width: 4px; border-radius: 2px;
    background: linear-gradient(to top, #10b981, #34d399);
    animation: wave 0.8s ease-in-out infinite alternate;
  }
  .wave-bar:nth-child(1){height:10px;animation-delay:0.0s;}
  .wave-bar:nth-child(2){height:20px;animation-delay:0.1s;}
  .wave-bar:nth-child(3){height:28px;animation-delay:0.2s;}
  .wave-bar:nth-child(4){height:20px;animation-delay:0.3s;}
  .wave-bar:nth-child(5){height:10px;animation-delay:0.4s;}
  @keyframes wave { from{transform:scaleY(0.3);} to{transform:scaleY(1);} }

  /* Pulse rings */
  .pulse-rings { position: absolute; display: flex; align-items: center; justify-content: center; z-index: 1; }
  .pulse-ring {
    position: absolute; border-radius: 50%;
    border: 2px solid rgba(239,68,68,0.5);
    animation: pulse-ring 1.6s ease-out infinite; opacity: 0;
  }
  .pulse-ring:nth-child(1){width:120px;height:120px;animation-delay:0s;}
  .pulse-ring:nth-child(2){width:145px;height:145px;animation-delay:0.5s;}
  .pulse-ring:nth-child(3){width:170px;height:170px;animation-delay:1.0s;}
  @keyframes pulse-ring { 0%{opacity:0.6;transform:scale(0.85);} 100%{opacity:0;transform:scale(1.2);} }

  /* Name label */
  .tile-label {
    position: absolute; bottom: 14px; left: 16px;
    display: flex; align-items: center; gap: 8px; z-index: 5;
  }
  .tile-name {
    font-size: 13px; font-weight: 600; color: #f1f5f9;
    background: rgba(0,0,0,0.55); backdrop-filter: blur(8px);
    padding: 4px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);
  }
  .tile-mic-indicator {
    width: 28px; height: 28px; border-radius: 8px;
    background: rgba(0,0,0,0.55); backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.08);
    display: flex; align-items: center; justify-content: center; font-size: 13px;
  }

  /* Spinner */
  .spinner {
    width: 26px; height: 26px; border-radius: 50%;
    border: 3px solid rgba(99,102,241,0.2); border-top-color: #6366f1;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Bottom controls ── */
  .meet-controls {
    flex-shrink: 0; min-height: 72px; padding: 10px 0 8px;
    display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px;
    background: rgba(255,255,255,0.03);
    border-top: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(16px); z-index: 10;
  }
  .ctrl-btn {
    width: 52px; height: 52px; border-radius: 14px; border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center; font-size: 22px;
    transition: all 0.2s ease; outline: none;
  }
  .ctrl-btn:not(.active):not(:disabled) {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); color: #94a3b8;
  }
  .ctrl-btn:not(.active):not(:disabled):hover { background: rgba(255,255,255,0.1); transform: translateY(-2px); }
  .ctrl-btn.active {
    background: linear-gradient(135deg, #ef4444, #f97316);
    border: 1px solid rgba(239,68,68,0.3); color: #fff;
    box-shadow: 0 6px 24px rgba(239,68,68,0.4);
    animation: mic-pulse 1s ease-in-out infinite;
  }
  @keyframes mic-pulse { 0%,100%{transform:scale(1);} 50%{transform:scale(0.95);} }
  .ctrl-btn:disabled { opacity: 0.35; cursor: not-allowed; transform: none !important; }

  .begin-btn {
    padding: 0 32px; height: 52px; border-radius: 14px; border: none;
    cursor: pointer; font-size: 15px; font-weight: 700;
    font-family: 'Inter', sans-serif; letter-spacing: 0.3px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6); color: #fff;
    box-shadow: 0 6px 24px rgba(99,102,241,0.4);
    transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1);
  }
  .begin-btn:hover { transform: translateY(-2px); box-shadow: 0 10px 32px rgba(99,102,241,0.55); }
  .begin-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
  .ctrl-hint {
    font-size: 11px; color: #475569;
    text-align: center; font-weight: 500;
  }

  /* ── Transcript sidebar ── */
  .transcript-sidebar {
    width: 320px; flex-shrink: 0; display: flex; flex-direction: column;
    border-left: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.015); backdrop-filter: blur(20px); overflow: hidden;
  }
  .ts-header {
    flex-shrink: 0; padding: 16px 20px; border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex; align-items: center; justify-content: space-between;
  }
  .ts-header h3 { font-size: 12px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 1px; }
  .ts-count { font-size: 11px; font-weight: 600; background: rgba(99,102,241,0.15); color: #a5b4fc; padding: 2px 8px; border-radius: 999px; }
  .ts-body {
    flex: 1; overflow-y: auto; padding: 14px 16px;
    display: flex; flex-direction: column; gap: 10px;
    scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.1) transparent;
  }
  .ts-body::-webkit-scrollbar { width: 4px; }
  .ts-body::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 2px; }

  .ts-empty {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; color: #2d3748; font-size: 13px; gap: 10px;
  }
  .ts-empty-icon { font-size: 32px; opacity: 0.3; }

  /* Message bubbles */
  .msg {
    padding: 10px 12px; border-radius: 10px;
    animation: msg-in 0.3s ease;
  }
  @keyframes msg-in { from{opacity:0;transform:translateY(8px);} to{opacity:1;transform:translateY(0);} }
  .msg.ai {
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.15);
    border-radius: 10px 10px 10px 2px;
  }
  .msg.user {
    background: rgba(16,185,129,0.07); border: 1px solid rgba(16,185,129,0.12);
    border-radius: 10px 10px 2px 10px; margin-left: 12px;
  }
  .msg-role { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }
  .msg.ai   .msg-role { color: #818cf8; }
  .msg.user .msg-role { color: #34d399; }
  .msg-text { font-size: 12px; line-height: 1.65; color: #94a3b8; }

  /* ── Editable user answer box ── */
  .pending-answer-wrap {
    border-radius: 10px; overflow: hidden;
    border: 1px solid rgba(16,185,129,0.3); background: rgba(16,185,129,0.05);
    animation: msg-in 0.3s ease;
    margin-left: 12px;
  }
  .pending-answer-label {
    padding: 7px 12px 0;
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.8px; color: #34d399;
  }
  .pending-answer-textarea {
    width: 100%; background: transparent; border: none; outline: none;
    color: #cbd5e1; font-size: 12px; line-height: 1.65;
    font-family: 'Inter', sans-serif;
    padding: 6px 12px 10px; resize: none; min-height: 64px;
  }
  .pending-answer-actions {
    display: flex; gap: 8px; padding: 0 12px 10px;
  }
  .submit-btn {
    flex: 1; padding: 7px 0; border-radius: 8px; border: none; cursor: pointer;
    font-size: 12px; font-weight: 700; font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #10b981, #059669); color: #fff;
    box-shadow: 0 4px 14px rgba(16,185,129,0.35);
    transition: all 0.2s ease;
  }
  .submit-btn:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(16,185,129,0.45); }
  .submit-btn:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }
  .cancel-btn {
    padding: 7px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);
    cursor: pointer; font-size: 12px; font-weight: 600; font-family: 'Inter', sans-serif;
    background: rgba(255,255,255,0.05); color: #64748b; transition: all 0.2s ease;
  }
  .cancel-btn:hover { background: rgba(255,255,255,0.09); color: #94a3b8; }
`;

// ─── Inject styles ─────────────────────────────────────────────────────────────
if (Platform.OS === 'web' && typeof document !== 'undefined') {
  const existing = document.getElementById('ib-styles');
  if (!existing) {
    const style = document.createElement('style');
    style.id = 'ib-styles';
    style.textContent = WEB_STYLES;
    document.head.appendChild(style);
  }
  document.title = 'InterviewBetter — AI Interview';
}

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  // Core state
  const [phase, setPhase] = useState('idle');
  // phases: idle | ai_speaking | ready | recording | transcribing | pending_submit | submitting

  const [sound, setSound] = useState(null);
  const [interviewStarted, setInterviewStarted] = useState(false);
  const [nativeRecording, setNativeRecording] = useState(null);

  // Transcript items — each is {role: 'ai'|'user', text: string, pending: bool}
  const [messages, setMessages] = useState([]);

  // Editable user answer draft
  const [draftAnswer, setDraftAnswer] = useState('');

  const webMediaRecorderRef = useRef(null);
  const webAudioChunksRef = useRef([]);
  const tsBodyRef = useRef(null);

  const scrollToBottom = () =>
    setTimeout(() => { if (tsBodyRef.current) tsBodyRef.current.scrollTop = tsBodyRef.current.scrollHeight; }, 80);

  const addMessage = (role, text, pending = false) => {
    setMessages(prev => [...prev, { role, text, pending }]);
    scrollToBottom();
  };

  // ── Status label derived from phase ────────────────────────────────────────
  const statusMap = {
    idle: 'Ready',
    ai_speaking: 'AI Speaking',
    ready: 'Your Turn',
    recording: 'Recording',
    transcribing: 'Transcribing…',
    pending_submit: 'Review Answer',
    submitting: 'Processing…',
  };
  const pillClass = () => {
    if (phase === 'ai_speaking') return 'speaking';
    if (phase === 'recording') return 'recording';
    if (['transcribing', 'submitting'].includes(phase)) return 'thinking';
    return '';
  };
  const statusLabel = statusMap[phase] || 'Ready';

  // ── 1. Begin Interview ───────────────────────────────────────────────────────
  async function beginInterview() {
    setPhase('submitting');
    try {
      const res = await fetch(`${SERVER_URL}/start-interview`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setInterviewStarted(true);
        await playAiResponse(data.ai_text || null);
      } else {
        setPhase('idle');
        alert('Server error starting interview.');
      }
    } catch (e) {
      console.error(e);
      setPhase('idle');
    }
  }

  // ── 2. Play AI audio + show AI transcript immediately when done ──────────────
  async function playAiResponse(aiTextHint) {
    setPhase('ai_speaking');
    try {
      // Fetch AI text if not provided
      let aiText = aiTextHint;
      if (!aiText) {
        try {
          const r = await fetch(`${SERVER_URL}/get-ai-transcript`);
          if (r.ok) { const d = await r.json(); aiText = d.ai_text; }
        } catch (_) { }
      }

      const { sound: playbackObj } = await Audio.Sound.createAsync(
        { uri: `${SERVER_URL}/get-audio?rnd=${Math.random()}` },
        { shouldPlay: true }
      );
      setSound(playbackObj);

      playbackObj.setOnPlaybackStatusUpdate((s) => {
        if (s.didJustFinish) {
          // ✅ Show AI transcript immediately after audio ends
          addMessage('ai', aiText || 'AI responded.');
          setPhase('ready');
        }
      });
    } catch (err) {
      console.error('Playback error:', err);
      setPhase('ready');
    }
  }

  // ── 3. Start Recording ───────────────────────────────────────────────────────
  async function startRecording() {
    try {
      if (sound) { await sound.unloadAsync(); setSound(null); }

      if (Platform.OS === 'web') {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus' : 'audio/webm';
        const mr = new MediaRecorder(stream, { mimeType });
        webAudioChunksRef.current = [];
        mr.ondataavailable = (e) => { if (e.data.size > 0) webAudioChunksRef.current.push(e.data); };
        webMediaRecorderRef.current = mr;
        mr.start(100);
      } else {
        const { status: perm } = await Audio.requestPermissionsAsync();
        if (perm !== 'granted') return;
        await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
        const { recording } = await Audio.Recording.createAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
        setNativeRecording(recording);
      }
      setPhase('recording');
    } catch (err) {
      console.error(err);
    }
  }

  // ── 4. Stop Recording → Transcribe only ─────────────────────────────────────
  async function stopRecording() {
    if (phase !== 'recording') return;
    setPhase('transcribing');

    if (Platform.OS === 'web') {
      const mr = webMediaRecorderRef.current;
      if (!mr) return;
      mr.stop();
      mr.stream.getTracks().forEach(t => t.stop());
      mr.onstop = async () => {
        const blob = new Blob(webAudioChunksRef.current, { type: mr.mimeType });
        await transcribeAudio(blob);
      };
    } else {
      if (!nativeRecording) return;
      await nativeRecording.stopAndUnloadAsync();
      const uri = nativeRecording.getURI();
      setNativeRecording(null);
      // native path: fetch blob from uri
      const resp = await fetch(uri);
      const blob = await resp.blob();
      await transcribeAudio(blob, uri);
    }
  }

  // ── 5. Transcribe audio → show editable draft ───────────────────────────────
  async function transcribeAudio(blob, nativeUri) {
    try {
      const formData = new FormData();
      if (nativeUri) {
        const ext = nativeUri.split('.').pop().toLowerCase();
        const mime = ext === 'wav' ? 'audio/wav' : ext === 'mp4' ? 'audio/mp4' : 'audio/m4a';
        formData.append('file', blob, `recording.${ext}`);
      } else {
        const ext = blob.type.includes('ogg') ? 'ogg' : blob.type.includes('mp4') ? 'mp4' : 'webm';
        formData.append('file', blob, `recording.${ext}`);
      }

      const res = await fetch(`${SERVER_URL}/transcribe-audio`, { method: 'POST', body: formData });
      if (res.ok) {
        const data = await res.json();
        const text = data.user_text || '';
        // ✅ Show editable transcript immediately
        setDraftAnswer(text);
        setPhase('pending_submit');
        scrollToBottom();
      } else {
        console.error('Transcription failed', res.status);
        setPhase('ready');
      }
    } catch (err) {
      console.error('Transcribe error:', err);
      setPhase('ready');
    }
  }

  // ── 6. Submit (possibly edited) answer ──────────────────────────────────────
  async function submitAnswer() {
    const text = draftAnswer.trim();
    setPhase('submitting');

    // Lock the user message in transcript
    addMessage('user', text || '(no answer)');
    setDraftAnswer('');

    try {
      const res = await fetch(`${SERVER_URL}/submit-answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_text: text }),
      });
      if (res.ok) {
        const data = await res.json();
        await playAiResponse(data.ai_text || null);
      } else {
        console.error('Submit failed', res.status);
        setPhase('ready');
      }
    } catch (err) {
      console.error('Submit error:', err);
      setPhase('ready');
    }
  }

  // ── 7. Re-record — discard draft ────────────────────────────────────────────
  function discardDraft() {
    setDraftAnswer('');
    setPhase('ready');
  }

  // ── Render ───────────────────────────────────────────────────────────────────
  if (Platform.OS !== 'web') return null;

  const isAiSpeaking = phase === 'ai_speaking';
  const isRecording = phase === 'recording';
  const isBusy = ['submitting', 'transcribing', 'ai_speaking'].includes(phase);
  const isPendingSubmit = phase === 'pending_submit';
  const totalMsgs = messages.length + (isPendingSubmit ? 1 : 0);

  return (
    <div className="meet-shell">

      {/* ── Top bar ── */}
      <header className="meet-topbar">
        <div className="topbar-brand">
          <div className="topbar-logo">🎓</div>
          <span className="topbar-title">InterviewBetter</span>
          <span className="topbar-sub">SST Mathematics Admission</span>
        </div>
        <div className={`status-pill ${pillClass()}`}>
          <div className="status-dot" />
          {statusLabel}
        </div>
        <div className="timer-label">
          {interviewStarted ? phase === 'ready' ? 'Your turn to answer' : statusLabel : 'Not started'}
        </div>
      </header>

      {/* ── Body ── */}
      <div className="meet-body">

        {/* ── Participants (left) ── */}
        <div className="participants-area">

          {/* AI tile */}
          <div className={`participant-tile ai-tile ${isAiSpeaking ? 'active' : ''}`}>
            <div className="tile-bg" />
            <div className="tile-avatar-wrap">
              <div className="tile-avatar">
                {isBusy && !isAiSpeaking
                  ? <div className="spinner" />
                  : <span>🤖</span>
                }
              </div>
              <div className={`wave-bars ${isAiSpeaking ? 'active' : ''}`}>
                {[1, 2, 3, 4, 5].map(i => <div key={i} className="wave-bar" />)}
              </div>
            </div>
            <div className="tile-label">
              <div className="tile-name">AI Interviewer</div>
              <div className="tile-mic-indicator">{isAiSpeaking ? '🔊' : '🔇'}</div>
            </div>
          </div>

          {/* User tile */}
          <div className={`participant-tile user-tile ${isRecording ? 'active' : ''}`}>
            <div className="tile-bg" />
            {isRecording && (
              <div className="pulse-rings">
                <div className="pulse-ring" />
                <div className="pulse-ring" />
                <div className="pulse-ring" />
              </div>
            )}
            <div className="tile-avatar-wrap">
              <div className="tile-avatar"><span>🧑‍💼</span></div>
            </div>
            <div className="tile-label">
              <div className="tile-name">You</div>
              <div className="tile-mic-indicator">{isRecording ? '🎙' : '🔇'}</div>
            </div>
          </div>

        </div>

        {/* ── Transcript sidebar ── */}
        <aside className="transcript-sidebar">
          <div className="ts-header">
            <h3>Transcript</h3>
            {totalMsgs > 0 && <span className="ts-count">{totalMsgs}</span>}
          </div>

          <div className="ts-body" ref={tsBodyRef}>
            {messages.length === 0 && !isPendingSubmit ? (
              <div className="ts-empty">
                <div className="ts-empty-icon">💬</div>
                <div>Transcript appears here as the interview progresses.</div>
              </div>
            ) : (
              <>
                {messages.map((m, i) => (
                  <div key={i} className={`msg ${m.role}`}>
                    <div className="msg-role">{m.role === 'ai' ? '🤖 Interviewer' : '🎙 You'}</div>
                    <div className="msg-text">{m.text}</div>
                  </div>
                ))}

                {/* Editable draft — shown after transcription, before submit */}
                {isPendingSubmit && (
                  <div className="pending-answer-wrap">
                    <div className="pending-answer-label">🎙 Your Answer — Review &amp; Edit</div>
                    <textarea
                      className="pending-answer-textarea"
                      value={draftAnswer}
                      onChange={e => setDraftAnswer(e.target.value)}
                      placeholder="Your transcribed answer will appear here…"
                      rows={3}
                    />
                    <div className="pending-answer-actions">
                      <button className="cancel-btn" onClick={discardDraft} title="Re-record">
                        🔄 Re-record
                      </button>
                      <button
                        className="submit-btn"
                        onClick={submitAnswer}
                        disabled={!draftAnswer.trim()}
                      >
                        Submit ✓
                      </button>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </aside>

      </div>

      {/* ── Bottom controls ── */}
      <div className="meet-controls">
        {!interviewStarted ? (
          <button
            className="begin-btn"
            onClick={beginInterview}
            disabled={phase === 'submitting'}
          >
            {phase === 'submitting' ? '⏳ Connecting…' : '✦ Begin Interview'}
          </button>
        ) : isPendingSubmit ? (
          <span style={{ fontSize: 13, color: '#475569' }}>
            ✏️ Edit your answer in the transcript, then click <strong style={{ color: '#10b981' }}>Submit</strong>
          </span>
        ) : (
          <button
            className={`ctrl-btn ${isRecording ? 'active' : ''}`}
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isBusy || isPendingSubmit}
            title={isRecording ? 'Stop & Transcribe' : 'Start Recording'}
          >
            {isRecording ? '⏹' : '🎙'}
          </button>
        )}
        {interviewStarted && !isPendingSubmit && (
          <div className="ctrl-hint">
            {isAiSpeaking ? 'AI is speaking…'
              : phase === 'transcribing' ? 'Transcribing your audio…'
                : phase === 'submitting' ? 'Generating AI response…'
                  : isRecording ? 'Tap to stop & transcribe'
                    : 'Tap mic to answer'}
          </div>
        )}
      </div>

    </div>
  );
}