import { useState, useRef, useEffect } from "react"

const API = import.meta.env.VITE_API_URL || "http://localhost:8000"

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0e0f11;
    --surface:  #16181c;
    --surface2: #1e2026;
    --border:   #2a2d35;
    --border2:  #363a45;
    --text:     #e8eaf0;
    --muted:    #7a7f8e;
    --accent:   #4f8ef7;
    --accent2:  #7c6af7;
    --green:    #3ecf8e;
    --amber:    #f0a050;
    --red:      #f06060;
    --mono:     'IBM Plex Mono', monospace;
    --sans:     'IBM Plex Sans', sans-serif;
  }

  body { background: var(--bg); color: var(--text); font-family: var(--sans); }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

  .layout { display: flex; height: 100vh; overflow: hidden; }

  /* Sidebar */
  .sidebar {
    width: 280px; flex-shrink: 0;
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
    overflow-y: auto;
  }
  .sidebar-header {
    padding: 20px 18px 16px;
    border-bottom: 1px solid var(--border);
  }
  .logo { font-size: 13px; font-weight: 600; letter-spacing: 0.08em; color: var(--text); text-transform: uppercase; }
  .logo span { color: var(--accent); }
  .sidebar-body { padding: 14px; flex: 1; display: flex; flex-direction: column; gap: 14px; }

  /* Sections */
  .section-label {
    font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 8px;
  }

  /* Upload */
  .upload-zone {
    border: 1px dashed var(--border2); border-radius: 8px;
    padding: 14px; text-align: center; cursor: pointer;
    transition: all 0.2s;
  }
  .upload-zone:hover { border-color: var(--accent); background: rgba(79,142,247,0.05); }
  .upload-btn {
    background: var(--accent); color: #fff; border: none;
    padding: 7px 14px; border-radius: 6px; font-size: 12px;
    font-weight: 500; cursor: pointer; font-family: var(--sans);
    transition: opacity 0.15s;
  }
  .upload-btn:hover { opacity: 0.85; }
  .upload-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .doc-list { margin-top: 8px; display: flex; flex-direction: column; gap: 3px; }
  .doc-item {
    font-size: 11px; color: var(--muted); padding: 4px 8px;
    background: var(--surface2); border-radius: 4px;
    display: flex; align-items: center; gap: 6px;
  }
  .doc-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--green); flex-shrink: 0; }

  /* Report panel */
  .report-panel {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px;
  }
  .report-select, .report-input {
    width: 100%; background: var(--bg); border: 1px solid var(--border2);
    border-radius: 6px; padding: 7px 10px; color: var(--text);
    font-size: 12px; font-family: var(--sans); margin-bottom: 8px;
    outline: none; transition: border-color 0.15s;
  }
  .report-select:focus, .report-input:focus { border-color: var(--accent); }
  .report-input::placeholder { color: var(--muted); }
  .gen-btn {
    width: 100%; background: var(--surface); border: 1px solid var(--border2);
    border-radius: 6px; padding: 8px; color: var(--text); font-size: 12px;
    font-weight: 500; cursor: pointer; font-family: var(--sans);
    transition: all 0.15s;
  }
  .gen-btn:hover:not(:disabled) { border-color: var(--accent2); color: var(--accent2); }
  .gen-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Mode toggle */
  .mode-btns { display: flex; gap: 6px; }
  .mode-btn {
    flex: 1; padding: 7px 10px; border-radius: 6px; font-size: 11px;
    font-weight: 500; cursor: pointer; font-family: var(--sans);
    border: 1px solid var(--border2); background: transparent;
    color: var(--muted); transition: all 0.15s;
  }
  .mode-btn.active { background: var(--surface2); border-color: var(--accent); color: var(--accent); }

  /* Main chat area */
  .main { flex: 1; display: flex; flex-direction: column; min-width: 0; }

  .chat-header {
    padding: 14px 24px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 10px; background: var(--surface);
  }
  .chat-header-title { font-size: 13px; font-weight: 500; color: var(--text); }
  .mode-badge {
    font-size: 10px; padding: 2px 8px; border-radius: 20px; font-weight: 600;
    letter-spacing: 0.06em;
  }
  .mode-badge.rag { background: rgba(62,207,142,0.12); color: var(--green); }
  .mode-badge.agent { background: rgba(124,106,247,0.12); color: var(--accent2); }

  /* Messages */
  .messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 20px; }

  .empty-state {
    flex: 1; display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 8px; color: var(--muted);
  }
  .empty-icon { font-size: 32px; opacity: 0.3; }
  .empty-text { font-size: 13px; }

  /* Message bubbles */
  .msg-row { display: flex; gap: 10px; align-items: flex-start; }
  .msg-row.user { flex-direction: row-reverse; }

  .msg-avatar {
    width: 28px; height: 28px; border-radius: 6px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 600;
  }
  .msg-avatar.user { background: rgba(79,142,247,0.2); color: var(--accent); }
  .msg-avatar.assistant { background: rgba(124,106,247,0.2); color: var(--accent2); }

  .msg-content { max-width: 78%; display: flex; flex-direction: column; gap: 8px; }

  .msg-bubble {
    padding: 12px 15px; border-radius: 10px; font-size: 13.5px;
    line-height: 1.65; border: 1px solid transparent;
  }
  .msg-bubble.user {
    background: rgba(79,142,247,0.1); border-color: rgba(79,142,247,0.2);
    color: var(--text);
  }
  .msg-bubble.assistant {
    background: var(--surface); border-color: var(--border);
    color: var(--text);
  }
  .msg-bubble.report {
    background: var(--surface); border-color: var(--border);
    font-family: var(--sans); font-size: 13px;
  }

  /* Formatted report content */
  .report-content h1 { font-size: 16px; font-weight: 600; color: var(--text); margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
  .report-content h2 { font-size: 13px; font-weight: 600; color: var(--accent); margin: 14px 0 6px; text-transform: uppercase; letter-spacing: 0.06em; }
  .report-content p { font-size: 13px; line-height: 1.65; color: var(--text); margin-bottom: 8px; }
  .report-content blockquote { border-left: 2px solid var(--accent2); padding-left: 10px; color: var(--muted); font-style: italic; margin-bottom: 10px; }
  .report-content ul { padding-left: 0; list-style: none; display: flex; flex-direction: column; gap: 5px; margin-bottom: 10px; }
  .report-content li { font-size: 13px; line-height: 1.6; color: var(--text); display: flex; gap: 8px; }
  .report-content li::before { content: '—'; color: var(--accent); flex-shrink: 0; }
  .report-content em { color: var(--muted); font-style: normal; font-size: 11px; }
  .report-content strong { font-weight: 600; color: var(--text); }
  .report-content table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 12px; }
  .report-content th { background: var(--surface2); padding: 7px 10px; text-align: left; font-weight: 600; border: 1px solid var(--border); color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
  .report-content td { padding: 7px 10px; border: 1px solid var(--border); line-height: 1.5; }
  .report-content tr:nth-child(even) td { background: rgba(255,255,255,0.02); }

  /* Source cards */
  .sources-header { font-size: 10px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
  .sources-grid { display: flex; flex-direction: column; gap: 5px; }
  .source-card {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 7px; padding: 9px 11px;
    display: flex; align-items: flex-start; gap: 10px;
  }
  .source-score {
    font-family: var(--mono); font-size: 11px; font-weight: 500;
    padding: 2px 6px; border-radius: 4px; flex-shrink: 0; margin-top: 1px;
  }
  .source-score.high { background: rgba(62,207,142,0.12); color: var(--green); }
  .source-score.mid  { background: rgba(240,160,80,0.12);  color: var(--amber); }
  .source-score.low  { background: rgba(240,96,96,0.12);   color: var(--red);   }
  .source-info { min-width: 0; flex: 1; }
  .source-name { font-size: 12px; font-weight: 500; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .source-preview { font-size: 11px; color: var(--muted); margin-top: 2px; line-height: 1.5; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }

  /* Tool call badges */
  .tool-calls { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 4px; }
  .tool-badge {
    font-size: 10px; font-family: var(--mono); padding: 2px 8px;
    border-radius: 4px; background: rgba(124,106,247,0.1);
    color: var(--accent2); border: 1px solid rgba(124,106,247,0.2);
  }

  /* Typing indicator */
  .typing { display: flex; gap: 4px; align-items: center; padding: 12px 15px; }
  .typing span {
    width: 5px; height: 5px; border-radius: 50%; background: var(--muted);
    animation: bounce 1.2s infinite;
  }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-5px)} }

  /* Input bar */
  .input-bar {
    padding: 14px 20px; border-top: 1px solid var(--border);
    background: var(--surface); display: flex; gap: 10px; align-items: flex-end;
  }
  .chat-input {
    flex: 1; background: var(--bg); border: 1px solid var(--border2);
    border-radius: 8px; padding: 10px 14px; color: var(--text);
    font-size: 13.5px; font-family: var(--sans); outline: none;
    resize: none; min-height: 42px; max-height: 120px;
    line-height: 1.5; transition: border-color 0.15s;
  }
  .chat-input:focus { border-color: var(--accent); }
  .chat-input::placeholder { color: var(--muted); }
  .send-btn {
    background: var(--accent); color: #fff; border: none;
    border-radius: 8px; padding: 10px 16px; font-size: 13px;
    font-weight: 500; cursor: pointer; font-family: var(--sans);
    transition: opacity 0.15s; white-space: nowrap; height: 42px;
  }
  .send-btn:hover:not(:disabled) { opacity: 0.85; }
  .send-btn:disabled { opacity: 0.35; cursor: not-allowed; }

  @keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
  .msg-row { animation: fadeIn 0.2s ease-out; }
`

// ── Render markdown-like report text into HTML ── //
function renderMarkdown(text) {
  if (!text) return ""
  const lines = text.split("\n")
  let html = ""
  let inTable = false
  let tableRows = []

  const flushTable = () => {
    if (!tableRows.length) return
    let t = "<table>"
    tableRows.forEach((row, i) => {
      const cells = row.split("|").map(c => c.trim()).filter(Boolean)
      if (i === 0) {
        t += "<tr>" + cells.map(c => `<th>${c}</th>`).join("") + "</tr>"
      } else if (row.match(/^[\s|:-]+$/)) {
        // skip separator row
      } else {
        t += "<tr>" + cells.map(c => `<td>${c}</td>`).join("") + "</tr>"
      }
    })
    t += "</table>"
    html += t
    tableRows = []
    inTable = false
  }

  lines.forEach(line => {
    if (line.startsWith("|")) {
      inTable = true
      tableRows.push(line)
      return
    }
    if (inTable) flushTable()

    if (line.startsWith("# "))       html += `<h1>${line.slice(2)}</h1>`
    else if (line.startsWith("## ")) html += `<h2>${line.slice(3)}</h2>`
    else if (line.startsWith("> "))  html += `<blockquote>${line.slice(2)}</blockquote>`
    else if (line.startsWith("- "))  html += `<ul><li>${formatInline(line.slice(2))}</li></ul>`
    else if (line.trim() === "")     html += ""
    else html += `<p>${formatInline(line)}</p>`
  })
  if (inTable) flushTable()

  // Merge consecutive <ul> items
  html = html.replace(/<\/ul><ul>/g, "")
  return html
}

function formatInline(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`(.*?)`/g, `<code style="font-family:IBM Plex Mono,monospace;font-size:11px;background:rgba(255,255,255,0.06);padding:1px 5px;border-radius:3px;">$1</code>`)
}

// ── Source card ── //
function SourceCard({ chunk }) {
  const pct  = Math.round((chunk.similarity || 0) * 100)
  const tier = pct >= 70 ? "high" : pct >= 50 ? "mid" : "low"
  return (
    <div className="source-card">
      <span className={`source-score ${tier}`}>{pct}%</span>
      <div className="source-info">
        <div className="source-name">{chunk.source}</div>
        {chunk.text && (
          <div className="source-preview">{chunk.text.slice(0, 140)}…</div>
        )}
      </div>
    </div>
  )
}

// ── Single message ── //
function Message({ msg }) {
  const isUser   = msg.role === "user"
  const isReport = msg.type === "report"

  return (
    <div className={`msg-row ${isUser ? "user" : ""}`}>
      <div className={`msg-avatar ${isUser ? "user" : "assistant"}`}>
        {isUser ? "YOU" : "AI"}
      </div>
      <div className="msg-content">
        {isReport ? (
          <div className="msg-bubble report">
            <div
              className="report-content"
              dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
            />
          </div>
        ) : (
          <div className={`msg-bubble ${isUser ? "user" : "assistant"}`}>
            {msg.content}
          </div>
        )}

        {/* Tool call badges */}
        {msg.tool_calls && msg.tool_calls.length > 0 && (
          <div className="tool-calls">
            {msg.tool_calls.map((t, i) => (
              <span key={i} className="tool-badge">{t}</span>
            ))}
          </div>
        )}

        {/* Source cards */}
        {msg.sources && msg.sources.length > 0 && (
          <div>
            <div className="sources-header">Sources used</div>
            <div className="sources-grid">
              {msg.sources.map((s, i) => (
                <SourceCard key={i} chunk={s} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Sidebar upload zone ── //
function UploadZone({ onUpload, docs }) {
  const inputRef   = useRef()
  const [busy, setBusy] = useState(false)

  const handleFiles = async (files) => {
    setBusy(true)
    const form = new FormData()
    Array.from(files).forEach(f => form.append("files", f))
    await fetch(`${API}/index`, { method: "POST", body: form })
    setBusy(false)
    onUpload()
  }

  return (
    <div>
      <div className="section-label">Documents</div>
      <div className="upload-zone" onClick={() => inputRef.current.click()}>
        <input
          ref={inputRef} type="file" multiple accept=".pdf,.txt,.md"
          style={{ display: "none" }}
          onChange={e => handleFiles(e.target.files)}
        />
        <button className="upload-btn" disabled={busy}>
          {busy ? "Indexing…" : "+ Upload documents"}
        </button>
      </div>
      {docs.length > 0 && (
        <div className="doc-list">
          {docs.map((d, i) => (
            <div key={i} className="doc-item">
              <span className="doc-dot" />
              {d}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Report generator panel ── //
function ReportPanel({ onGenerate, loading }) {
  const [type,   setType]   = useState("bullet_summary")
  const [topic,  setTopic]  = useState("")
  const [topicB, setTopicB] = useState("")

  const labels = {
    bullet_summary: "Bullet summary",
    research_brief: "Research brief",
    comparison:     "Comparison"
  }

  return (
    <div>
      <div className="section-label">Generate report</div>
      <div className="report-panel">
        <select
          className="report-select"
          value={type}
          onChange={e => setType(e.target.value)}
        >
          {Object.entries(labels).map(([v, l]) => (
            <option key={v} value={v}>{l}</option>
          ))}
        </select>

        <input
          className="report-input"
          placeholder={type === "comparison" ? "Topic A" : "Topic or question"}
          value={topic}
          onChange={e => setTopic(e.target.value)}
        />

        {type === "comparison" && (
          <input
            className="report-input"
            placeholder="Topic B"
            value={topicB}
            onChange={e => setTopicB(e.target.value)}
          />
        )}

        <button
          className="gen-btn"
          onClick={() => { onGenerate(type, topic, topicB); setTopic(""); setTopicB("") }}
          disabled={!topic.trim() || loading}
        >
          {loading ? "Generating…" : `Generate ${labels[type]} ↗`}
        </button>
      </div>
    </div>
  )
}

// ── Main app ── //
export default function App() {
  const [messages, setMessages] = useState([])
  const [input,    setInput]    = useState("")
  const [loading,  setLoading]  = useState(false)
  const [mode,     setMode]     = useState("rag")
  const [docs,     setDocs]     = useState([])
  const bottomRef  = useRef()
  const inputRef   = useRef()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, loading])

  useEffect(() => {
    fetch(`${API}/docs`).then(r => r.json()).then(d => setDocs(d.docs || []))
  }, [])

  const addMsg = (role, content, extras = {}) =>
    setMessages(prev => [...prev, { role, content, ...extras }])

  const sendMessage = async () => {
    if (!input.trim() || loading) return
    const q = input.trim()
    setInput("")
    addMsg("user", q)
    setLoading(true)

    try {
      const endpoint = mode === "agent" ? "/agent" : "/ask"
      const res  = await fetch(`${API}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q })
      })
      const data = await res.json()
      addMsg("assistant", data.answer, {
        sources:    data.sources    || [],
        tool_calls: data.tool_calls || []
      })
    } catch (e) {
      addMsg("assistant", `Connection error: ${e.message}. Is the server running?`)
    }
    setLoading(false)
  }

  const generateReport = async (type, topic, topicB) => {
    addMsg("user", `Generate ${type.replace(/_/g, " ")}: "${topic}"${topicB ? ` vs "${topicB}"` : ""}`)
    setLoading(true)
    try {
      const res  = await fetch(`${API}/report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type, topic, topic_b: topicB })
      })
      const data = await res.json()
      addMsg("assistant", data.rendered, { type: "report" })
    } catch (e) {
      addMsg("assistant", `Error: ${e.message}`)
    }
    setLoading(false)
  }

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <>
      <style>{styles}</style>
      <div className="layout">

        {/* Sidebar */}
        <div className="sidebar">
          <div className="sidebar-header">
            <div className="logo">Research<span>AI</span></div>
          </div>
          <div className="sidebar-body">
            <UploadZone
              docs={docs}
              onUpload={() =>
                fetch(`${API}/docs`).then(r => r.json()).then(d => setDocs(d.docs || []))
              }
            />

            <ReportPanel onGenerate={generateReport} loading={loading} />

            <div>
              <div className="section-label">Answer mode</div>
              <div className="mode-btns">
                {["rag", "agent"].map(m => (
                  <button
                    key={m}
                    className={`mode-btn ${mode === m ? "active" : ""}`}
                    onClick={() => setMode(m)}
                  >
                    {m === "rag" ? "RAG fast" : "Agent deep"}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Main */}
        <div className="main">
          <div className="chat-header">
            <div className="chat-header-title">Research Assistant</div>
            <span className={`mode-badge ${mode}`}>
              {mode === "rag" ? "RAG mode" : "Agent mode"}
            </span>
            <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--muted)" }}>
              {docs.length} doc{docs.length !== 1 ? "s" : ""} indexed
            </span>
          </div>

          <div className="messages">
            {messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">⬡</div>
                <div className="empty-text">Upload documents, then ask anything.</div>
              </div>
            ) : (
              messages.map((m, i) => <Message key={i} msg={m} />)
            )}

            {loading && (
              <div className="msg-row">
                <div className="msg-avatar assistant">AI</div>
                <div className="msg-bubble assistant">
                  <div className="typing">
                    <span /><span /><span />
                  </div>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="input-bar">
            <textarea
              ref={inputRef}
              className="chat-input"
              placeholder="Ask a question about your documents…  (Enter to send, Shift+Enter for new line)"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              rows={1}
            />
            <button className="send-btn" onClick={sendMessage} disabled={loading || !input.trim()}>
              Send
            </button>
          </div>
        </div>
      </div>
    </>
  )
}