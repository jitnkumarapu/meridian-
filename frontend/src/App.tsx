import { ArrowUp, ChartNoAxesCombined, Check, ChevronDown, Clock3, FileText, Library, Menu, Moon, Plus, Search, Settings2, Sun, Trash2, Upload, X } from 'lucide-react'
import { FormEvent, useEffect, useRef, useState } from 'react'
import { EvidencePanel, ReadableAnswer, CitationChip, TableRenderer } from './components'
import type { Answer, DocumentDetail, DocumentRecord, EvaluationRun, Evidence, HistoryEntry, JobRecord, PageMetadata, PageName, RuntimeSettings } from './types'
import './style.css'

const defaultSettings: RuntimeSettings = { final_k: 5, candidate_k: 20, fusion: 'rrf', rrf_k: 60, dense_weight: .6, sparse_weight: .4, provider: 'mock', model: 'deterministic-evidence-v1', vision: false, demo_mode: true }
const examples = ['What was FY2024 revenue?', 'Which segment grew fastest?', 'What trend does the revenue chart show?']

async function jsonFetch<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options)
  if (!response.ok) throw new Error(await response.text())
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

function cleanText(value: string) {
  try {
    return value.replaceAll('â', '’').replaceAll('â', '–').replaceAll('â', '—').replaceAll('Â₹', '₹')
      .replaceAll('â¹', '₹')
  } catch {
    return value
  }
}

export default function App() {
  const [page, setPage] = useState<PageName>('search')
  const [query, setQuery] = useState('')
  const [answer, setAnswer] = useState<Answer>()
  const [loading, setLoading] = useState(false)
  const [dark, setDark] = useState(false)
  const [selected, setSelected] = useState<Evidence>()
  const [error, setError] = useState('')
  const [documents, setDocuments] = useState<DocumentRecord[]>([])
  const [included, setIncluded] = useState<Record<string, boolean>>(() => JSON.parse(localStorage.getItem('meridian-included-documents') || '{}') as Record<string, boolean>)
  const [jobs, setJobs] = useState<JobRecord[]>([])
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [evaluations, setEvaluations] = useState<EvaluationRun[]>([])
  const [evaluation, setEvaluation] = useState<EvaluationRun>()
  const [settings, setSettings] = useState<RuntimeSettings>(defaultSettings)
  const [scope, setScope] = useState('')
  const [answerStyle, setAnswerStyle] = useState<'compact' | 'detailed'>('compact')
  const [detail, setDetail] = useState<DocumentDetail>()
  const [conversationId, setConversationId] = useState<string>(() => crypto.randomUUID())
  const [isMobile, setIsMobile] = useState(() => typeof window !== 'undefined' && window.matchMedia?.('(max-width: 1023px)').matches)
  const request = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!window.matchMedia) return
    const media = window.matchMedia('(max-width: 1023px)')
    const update = () => setIsMobile(media.matches)
    update()
    media.addEventListener?.('change', update)
    return () => media.removeEventListener?.('change', update)
  }, [])

  const loadDocuments = () => Promise.all([jsonFetch<DocumentRecord[]>('/api/documents'), jsonFetch<JobRecord[]>('/api/jobs')])
    .then(([nextDocuments, nextJobs]) => { setDocuments(nextDocuments); setJobs(nextJobs) })

  useEffect(() => {
    setError('')
    if (page === 'library' || page === 'search') loadDocuments().catch(() => setError('Documents are unavailable right now.'))
    if (page === 'history') jsonFetch<HistoryEntry[]>('/api/history').then(setHistory).catch(() => setError('Recent questions are unavailable right now.'))
    if (page === 'evaluations') jsonFetch<EvaluationRun[]>('/api/evaluations').then(runs => {
      setEvaluations(runs)
      if (runs[0]) jsonFetch<EvaluationRun>(`/api/evaluations/${runs[0].run_id}`).then(setEvaluation)
    }).catch(() => setError('The quality report is unavailable right now.'))
    if (page === 'settings') jsonFetch<RuntimeSettings>('/api/settings').then(setSettings).catch(() => setError('Settings are unavailable right now.'))
  }, [page])

  useEffect(() => {
    if (page !== 'library' || !jobs.some(job => ['queued', 'processing'].includes(job.state))) return
    const timer = window.setInterval(() => loadDocuments().catch(() => undefined), 2000)
    return () => window.clearInterval(timer)
  }, [page, jobs])

  async function ask(event?: FormEvent) {
    event?.preventDefault()
    if (!query.trim() || loading) return
    request.current?.abort()
    request.current = new AbortController()
    setLoading(true)
    setError('')
    try {
      const next = await jsonFetch<Answer>('/api/ask', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        signal: request.current.signal,
        body: JSON.stringify({ query: query.trim(), k: settings.final_k, doc_ids: scope ? [scope] : activeDocuments.map(document => document.doc_id), style: answerStyle, request_id: crypto.randomUUID(), conversation_id: conversationId }),
      })
      setAnswer(next)
      const nextConversationId = next.metadata?.conversation_id
      if (typeof nextConversationId === 'string') setConversationId(nextConversationId)
    } catch (caught) {
      if ((caught as Error).name !== 'AbortError') setError(caught instanceof Error ? caught.message : 'The answer could not be generated.')
    } finally {
      setLoading(false)
    }
  }

  async function upload(file?: File) {
    if (!file) return
    if (!['application/pdf', 'image/png', 'image/jpeg', 'image/tiff'].includes(file.type)) {
      setError('Please choose a PDF or a PNG, JPEG, or TIFF scan.')
      return
    }
    try {
      const job = await jsonFetch<JobRecord>(`/api/jobs/upload?filename=${encodeURIComponent(file.name)}`, { method: 'POST', headers: { 'content-type': file.type }, body: file })
      setJobs(previous => [job, ...previous])
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Upload failed. Check that the file is readable and try again.')
    }
  }

  function navigate(next: PageName) {
    setPage(next)
    setError('')
  }

  function toggleIncluded(docId: string) {
    setIncluded(previous => {
      const next = { ...previous, [docId]: previous[docId] === false }
      localStorage.setItem('meridian-included-documents', JSON.stringify(next))
      return next
    })
  }

  const nav: [PageName, string, typeof Search][] = [['search', 'Ask', Search], ['library', 'Documents', Library], ['evaluations', 'Quality report', ChartNoAxesCombined], ['settings', 'Settings', Settings2]]
  const activeDocuments = documents.filter(document => included[document.doc_id] !== false)
  return <main className={dark ? 'dark' : ''}>
    <h1 className="sr-only">Meridian evidence intelligence</h1>
    <aside className="sidebar">
      <button className="brand" onClick={() => navigate('search')}><span className="mark">M</span><span>Meridian<small>evidence intelligence</small></span></button>
      <button className="new-question" onClick={() => { setAnswer(undefined); setQuery(''); navigate('search') }}><Plus size={16} /> New question</button>
      <div className="sidebar-history"><div className="sidebar-label">Recent</div>{history.slice(0, 5).map(item => <button key={item.history_id} onClick={() => { setQuery(item.question); navigate('search') }}>{item.title || item.question}</button>)}</div>
      <div className="sidebar-nav">{nav.map(([key, label, Icon]) => <button key={key} aria-label={`${label}${label === 'Documents' && !isMobile ? ' / Library' : label === 'Quality report' ? ' / Evaluations' : ''}`} aria-current={page === key ? 'page' : undefined} onClick={() => navigate(key)}><Icon size={18} /><span>{label}</span></button>)}<button aria-label="History" aria-current={page === 'history' ? 'page' : undefined} onClick={() => navigate('history')}><Clock3 size={18} /><span>History</span></button></div>
    </aside>
    <nav className="mobile-nav" aria-label="Primary navigation"><button className="mobile-menu" aria-label="Open menu"><Menu size={20} /></button><button className="brand" onClick={() => navigate('search')}><span className="mark">M</span><span>Meridian</span></button><button aria-label={isMobile ? 'Documents / Library' : 'Mobile documents'} onClick={() => navigate('library')}><Library size={18} /></button></nav>
    <button className="theme-toggle" aria-label="Toggle theme" onClick={() => setDark(previous => !previous)}>{dark ? <Sun size={18} /> : <Moon size={18} />}</button>
    <section className="app-content">
    {error && <div className="error" role="alert">{error}</div>}
    {page === 'search' && <AskPage query={query} setQuery={setQuery} ask={ask} loading={loading} cancel={() => request.current?.abort()} examples={examples} documents={activeDocuments} scope={scope} setScope={setScope} answer={answer} open={setSelected} recent={history} goHistory={() => navigate('history')} upload={upload} />}
    {page === 'library' && <DocumentsPage documents={documents} jobs={jobs} included={included} toggleIncluded={toggleIncluded} upload={upload} open={async id => setDetail(await jsonFetch<DocumentDetail>(`/api/documents/${id}`))} refresh={loadDocuments} remove={async id => { if (!window.confirm('Remove this document from the search index?')) return; await jsonFetch(`/api/documents/${id}?confirm=true`, { method: 'DELETE' }); await loadDocuments() }} />}
    {page === 'history' && <HistoryPage entries={history} rerun={question => { setQuery(question); navigate('search') }} />}
    {page === 'evaluations' && <QualityPage runs={evaluations} selected={evaluation} choose={id => jsonFetch<EvaluationRun>(`/api/evaluations/${id}`).then(setEvaluation)} />}
    {page === 'settings' && <SettingsPage value={settings} answerStyle={answerStyle} setAnswerStyle={setAnswerStyle} save={async next => setSettings(await jsonFetch<RuntimeSettings>('/api/settings', { method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify(next) }))} />}
    {selected && <EvidenceDrawer evidence={selected} onClose={() => setSelected(undefined)} />}
    {detail && <ReviewDrawer detail={detail} onClose={() => setDetail(undefined)} />}
    <footer>Grounded in your documents · Not financial advice</footer>
    </section>
  </main>
}

function AskPage({ query, setQuery, ask, loading, cancel, examples, documents, scope, setScope, answer, open, recent, goHistory, upload }: { query: string; setQuery: (value: string) => void; ask: (event?: FormEvent) => void; loading: boolean; cancel: () => void; examples: string[]; documents: DocumentRecord[]; scope: string; setScope: (value: string) => void; answer?: Answer; open: (e: Evidence) => void; recent: HistoryEntry[]; goHistory: () => void; upload: (file?: File) => void }) {
  const input = useRef<HTMLInputElement>(null)
  const hasDocuments = documents.length > 0
  const [scopeOpen, setScopeOpen] = useState(false)
  return <><header className={answer ? 'conversation-header' : ''}><div className="kicker">{answer ? 'A QUESTION ABOUT YOUR DOCUMENTS' : 'YOUR DOCUMENTS, MADE CLEAR'}</div><h1>{answer ? 'Ask a follow-up.' : <>What would you like<br /><em>to know?</em></>}</h1><p>Ask in plain language. I’ll explain what the documents say and show you where it came from.</p><form className="composer" onSubmit={ask}><textarea rows={1} value={query} onChange={event => setQuery(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); ask() } }} placeholder="Ask about revenue, risks, segments, or trends…" aria-label="Question" disabled={!hasDocuments} /><div className="composer-toolbar">{hasDocuments ? <div className="scope-wrap"><button type="button" className="scope-chip" aria-label="Document scope" onClick={() => setScopeOpen(open => !open)}><FileText size={14} />{scope ? documents.find(document => document.doc_id === scope)?.company || scope : 'All documents'}<ChevronDown size={13} /></button>{scopeOpen && <div className="scope-menu" role="menu"><button type="button" onClick={() => { setScope(''); setScopeOpen(false) }}><Check size={14} /> All documents</button>{documents.map(document => <button type="button" role="menuitem" key={document.doc_id} onClick={() => { setScope(document.doc_id); setScopeOpen(false) }}><FileText size={14} /> {document.company || document.filename || document.doc_id}</button>)}</div>}</div> : <span className="hint">Add a document to get started</span>}<button aria-label="Ask" className="send-button" type="submit" disabled={!hasDocuments || loading}>{loading ? <X size={17} /> : <ArrowUp size={18} />}</button></div></form>{hasDocuments && !answer && <><div className="starter-label">Not sure where to start?</div><div className="examples">{examples.map(example => <button key={example} onClick={() => { setQuery(example); ask() }}>{example}<ArrowUp size={14} /></button>)}</div></>}{!hasDocuments && <div className="empty ask-empty"><FileText size={28} /><strong>Add a financial report and I’ll help you make sense of it.</strong><span>Meridian searches text, tables, and figures and cites every answer.</span><button className="upload" onClick={() => input.current?.click()}>Add a document</button><input ref={input} type="file" accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff" onChange={event => upload(event.target.files?.[0])} /></div>}</header><section className="workspace">{loading && <div className="loading-card" role="status"><span className="pulse-dot" /><strong>Reading your documents…</strong><span>Finding the passages, tables, and figures that matter.</span><button className="secondary" onClick={cancel}>Stop</button></div>}{answer && <AnswerView answer={answer} open={open} />}{!answer && !loading && hasDocuments && <div className="empty ask-empty"><FileText size={28} /><strong>Your answer will appear here</strong><span>Every important claim includes a document and page citation.</span></div>}{recent.length > 0 && <section className="recent"><div className="section-heading"><div><div className="kicker">RECENT QUESTIONS</div><h3>Continue where you left off</h3></div><button className="secondary" onClick={goHistory}>History</button></div>{recent.slice(0, 3).map(item => <button className="recent-item" key={item.history_id} onClick={() => setQuery(item.question)}>{item.title || item.question}</button>)}</section>}</section></>
}

function AnswerView({ answer, open }: { answer: Answer; open: (e: Evidence) => void }) {
  return <div className="answer-stack"><article className="answer"><p className="answer-context">From the documents you selected</p><ReadableAnswer text={answer.text} /><InlineEvidence answer={answer} open={open} /><div className="citation-list">{answer.citations.slice(0, 4).map((citation, index) => <CitationChip key={`${citation.doc_id}-${citation.page}-${index}`} doc={`${index + 1}. ${citation.doc_id}`} page={citation.page} modality={citation.modality} onClick={() => { const evidence = answer.evidence.find(item => item.chunk.chunk_id === citation.chunk_id); if (evidence) open(evidence) }} />)}{answer.citations.length > 4 && <span className="more-sources">+{answer.citations.length - 4} more</span>}</div><small>{answer.latency_ms} ms · {answer.evidence.length} supporting items</small></article><EvidencePanel items={answer.evidence} onOpen={open} /></div>
}

function InlineEvidence({ answer, open }: { answer: Answer; open: (e: Evidence) => void }) {
  const items = answer.evidence.filter(item => item.chunk.modality !== 'text').slice(0, 2)
  if (!items.length) return null
  return <section className="inline-evidence" aria-label="Supporting tables and figures"><div className="inline-evidence-title">Supporting material</div>{items.map(item => <article key={item.chunk.chunk_id} className="inline-evidence-item">{item.chunk.modality === 'table' && item.chunk.raw_grid ? <TableRenderer grid={item.chunk.raw_grid} /> : item.chunk.modality === 'figure' && <>{item.chunk.image_path && <img className="figure-preview" src={`/api/figure/${item.chunk.chunk_id}`} alt={item.chunk.caption || `Figure on page ${item.chunk.page}`} />}<p>{item.chunk.caption || 'Figure from the report'}</p></>}<button className="source-link" onClick={() => open(item)}>View page {item.chunk.page}</button></article>)}</section>
}

function EvidenceDrawer({ evidence, onClose }: { evidence: Evidence; onClose: () => void }) {
  const [metadata, setMetadata] = useState<PageMetadata>()
  useEffect(() => {
    jsonFetch<PageMetadata>(`/api/page/${evidence.chunk.doc_id}/${evidence.chunk.page}/metadata`).then(setMetadata).catch(() => undefined)
  }, [evidence])
  const box = evidence.chunk.bbox
  const width = metadata?.width || 1
  const height = metadata?.height || 1
  return <aside className="drawer review-drawer" role="dialog" aria-label="Source evidence"><button className="close" onClick={onClose}>Close</button><div className="kicker">SOURCE EVIDENCE</div><h2>{evidence.chunk.doc_id}, page {evidence.chunk.page}</h2><div className="page-preview"><div className="page-canvas"><img src={`/api/page/${evidence.chunk.doc_id}/${evidence.chunk.page}.png`} alt={`Page ${evidence.chunk.page} from ${evidence.chunk.doc_id}`} />{box && <div className="region" title="Extracted evidence region" style={{ left: `${box.x0 / width * 100}%`, top: `${box.y0 / height * 100}%`, width: `${(box.x1 - box.x0) / width * 100}%`, height: `${(box.y1 - box.y0) / height * 100}%` }} />}</div></div><p>{cleanText(evidence.chunk.content || evidence.chunk.caption || 'No text excerpt was extracted.')}</p></aside>
}

function DocumentsPage({ documents, jobs, included, toggleIncluded, upload, open, refresh, remove }: { documents: DocumentRecord[]; jobs: JobRecord[]; included: Record<string, boolean>; toggleIncluded: (id: string) => void; upload: (file?: File) => void; open: (id: string) => void; refresh: () => Promise<unknown>; remove: (id: string) => Promise<void> }) {
  const input = useRef<HTMLInputElement>(null)
  return <section className="page"><div className="page-heading"><div><div className="kicker">YOUR SOURCES</div><h2>Documents</h2><p>The files the assistant can answer questions from.</p></div><button className="upload" onClick={() => input.current?.click()}><Upload size={17} /> Upload</button><input ref={input} type="file" accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff" onChange={event => upload(event.target.files?.[0])} /></div><div className="drop-zone" onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); upload(event.dataTransfer.files[0]) }}>Drop a PDF or scan here, or use Upload. Files are parsed, indexed, and available to ask about when processing finishes.</div>{jobs.filter(job => ['queued', 'processing'].includes(job.state)).map(job => <article className="job" key={job.job_id}><strong>{job.filename}</strong><span>{job.stage.replaceAll('_', ' ')} · {job.progress}%</span><progress value={job.progress} max="100" /></article>)}{documents.length ? <div className="list">{documents.map(document => <article key={document.doc_id}><div className="row"><div><h3>{document.company || document.filename || document.doc_id}</h3><span>{document.filename} · {document.fiscal_year || 'Report'}</span></div><label className="include-toggle"><input type="checkbox" checked={included[document.doc_id] !== false} onChange={() => toggleIncluded(document.doc_id)} /> Include in search</label><span className={`status ${document.processing_state}`}>{document.processing_state || 'ready'}</span></div><div className="document-stats">{document.pages || 0} pages · {document.text_chunks || 0} text sections · {document.tables || 0} tables · {document.figures || 0} figures</div>{document.warning_count ? <div className="notice">{document.warning_count} extraction warning{document.warning_count === 1 ? '' : 's'}{document.warnings?.[0] ? `: ${document.warnings[0]}` : ''}</div> : null}<div className="actions"><button onClick={() => open(document.doc_id)}>Preview</button><button className="secondary" onClick={() => remove(document.doc_id)}><Trash2 size={15} /> Remove</button></div></article>)}</div> : <div className="empty"><FileText size={28} /><strong>No documents yet</strong><span>Upload a financial report to get started.</span></div>}<button className="secondary" onClick={() => refresh()}>Refresh</button></section>
}

function HistoryPage({ entries, rerun }: { entries: HistoryEntry[]; rerun: (question: string) => void }) {
  return <section className="page"><div className="kicker">RECENT QUESTIONS</div><h2>History</h2><p>Return to questions you have already asked.</p>{entries.length ? <div className="list">{entries.map(entry => <article key={entry.history_id}><div className="row"><strong>{entry.title || entry.question}</strong><span>{new Date(entry.created_at).toLocaleString()}</span></div><p>{entry.question}</p><button onClick={() => rerun(entry.question)}>Ask again</button></article>)}</div> : <div className="empty">No saved questions yet.</div>}</section>
}

function QualityPage({ runs, selected, choose }: { runs: EvaluationRun[]; selected?: EvaluationRun; choose: (id: string) => void }) {
  const systems = (selected?.metrics as { systems?: Record<string, Record<string, unknown>> } | undefined)?.systems
  const metrics = systems?.multimodal || (selected?.metrics || {}) as Record<string, unknown>
  const findNumber = (name: string) => { const value = metrics[name]; if (typeof value === 'number') return value; if (value && typeof value === 'object' && 'mean' in value && typeof value.mean === 'number') return value.mean; return undefined }
  return <section className="page"><div className="kicker">QUALITY REPORT</div><h2>How well Meridian performs</h2><p>These results come from the saved golden-set evaluation. Higher recall means the right evidence was found; lower latency means answers arrive faster.</p>{runs.length > 0 && <select aria-label="Evaluation run" value={selected?.run_id || runs[0].run_id} onChange={event => choose(event.target.value)}>{runs.map((run, index) => <option key={run.run_id} value={run.run_id}>{index === 0 ? 'Latest saved run' : `Saved run ${runs.length - index}`}</option>)}</select>}<div className="metric-grid"><Metric label="Recall at 10" value={findNumber('recall@10')} help="How often the relevant source appeared in the first ten results." /><Metric label="Recall at 5" value={findNumber('recall@5')} help="How often the relevant source appeared near the top." /><Metric label="Average latency" value={findNumber('latency_ms')} suffix=" ms" help="Average time spent retrieving evidence." /><Metric label="First relevant source" value={findNumber('mrr')} help="How close the first relevant source was to the top." /></div><h3>Failure explorer</h3>{selected?.per_question?.length ? <details><summary>Question-level results ({selected.per_question.length})</summary><pre>{JSON.stringify(selected.per_question, null, 2)}</pre></details> : <div className="notice">No detailed question results are included in this saved run.</div>}</section>
}

function Metric({ label, value, suffix = '', help }: { label: string; value?: number; suffix?: string; help: string }) {
  return <article><span className="kicker">{label}</span><strong className="metric-value">{value === undefined ? '—' : `${(value * (label.includes('Recall') || label.includes('rank') ? 100 : 1)).toFixed(label.includes('latency') ? 0 : 1)}${label.includes('Recall') || label.includes('rank') ? '%' : suffix}`}</strong><p>{help}</p></article>
}

function SettingsPage({ value, answerStyle, setAnswerStyle, save }: { value: RuntimeSettings; answerStyle: 'compact' | 'detailed'; setAnswerStyle: (style: 'compact' | 'detailed') => void; save: (next: RuntimeSettings) => Promise<void> }) {
  const [draft, setDraft] = useState(value)
  const [saved, setSaved] = useState(false)
  useEffect(() => setDraft(value), [value])
  async function submit() { await save(draft); setSaved(true); window.setTimeout(() => setSaved(false), 2200) }
  return <section className="page"><div className="kicker">YOUR PREFERENCES</div><h2>Settings</h2><p>Choose how much evidence and explanation you want in each answer.</p><div className="settings-grid"><label>Answer detail<select value={answerStyle} onChange={event => setAnswerStyle(event.target.value as 'compact' | 'detailed')}><option value="compact">Concise</option><option value="detailed">Detailed</option></select><small>Detailed answers explain more context around each figure and comparison.</small></label><label>Sources per answer<input type="number" min="1" max="20" value={draft.final_k} onChange={event => setDraft({ ...draft, final_k: Number(event.target.value) })} /><small>Increase this when a question needs more supporting evidence.</small></label></div><details><summary>Advanced (for developers)</summary><div className="settings-grid"><label>Final results K<input aria-label="Final results K" type="number" value={draft.final_k} onChange={event => setDraft({ ...draft, final_k: Number(event.target.value) })} /></label><label>Provider<input value={draft.provider} onChange={event => setDraft({ ...draft, provider: event.target.value })} /></label><label>Model<input value={draft.model} onChange={event => setDraft({ ...draft, model: event.target.value })} /></label></div></details><div className="actions"><button onClick={submit}>Save settings</button><button className="secondary" onClick={() => { setDraft(defaultSettings); setAnswerStyle('compact') }}>Reset to defaults</button>{saved && <span className="grounded" role="status">Settings saved.</span>}</div></section>
}

function ReviewDrawer({ detail, onClose }: { detail: DocumentDetail; onClose: () => void }) {
  return <aside className="drawer review-drawer" role="dialog" aria-label="Document preview"><button className="close" onClick={onClose}>Close</button><div className="kicker">DOCUMENT PREVIEW</div><h2>{detail.document.company || detail.document.filename}</h2><p className="drawer-subtitle">{detail.document.pages} pages · {detail.tables.length} tables · {detail.figures.length} figures</p><h3>Extracted figures</h3>{detail.figures.slice(0, 6).map(figure => <article className="evidence" key={figure.chunk_id}><div className="eyebrow">Page {figure.page}</div>{figure.image_path && <img className="figure-preview" src={`/api/figure/${figure.chunk_id}`} alt={figure.caption || `Figure on page ${figure.page}`} />}<p>{figure.caption || figure.content}</p></article>)}</aside>
}
