import { Fragment, useState } from 'react'
import type { ReactNode } from 'react'
import type { Evidence, Modality } from './types'

function cleanText(value:string) {
  return value
    .replaceAll('â', '’')
    .replaceAll('â', '–')
    .replaceAll('â', '—')
    .replaceAll('â', '‘')
    .replaceAll('â', '“')
    .replaceAll('â', '”')
    .replaceAll('Â₹', '₹')
    .replaceAll('â¹', '₹')
    .replace(/\s*\[[^\]]+,\s*p\.\d+,\s*(?:text|table|figure)\]\s*/g, ' ')
}

export function ReadableAnswer({text}:{text:string}) {
  const lines = cleanText(text).replace(/\r/g, '').split('\n')
  const blocks: ReactNode[] = []
  let paragraph: string[] = []
  let bullets: string[] = []
  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push(<p key={`p-${blocks.length}`}>{renderInline(paragraph.join(' '))}</p>)
      paragraph = []
    }
  }
  const flushBullets = () => {
    if (bullets.length) {
      blocks.push(<ul key={`ul-${blocks.length}`}>{bullets.map((item, index) => <li key={`${item}-${index}`}>{renderInline(item)}</li>)}</ul>)
      bullets = []
    }
  }
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index].trim()
    if (!line) { flushParagraph(); flushBullets(); continue }
    if (/^\|.*\|$/.test(line) && index + 1 < lines.length && /^\|?[\s|:-]+\|$/.test(lines[index + 1].trim())) {
      flushParagraph(); flushBullets()
      const header = splitTableRow(line)
      const rows: string[][] = []
      index += 2
      while (index < lines.length && /^\|.*\|$/.test(lines[index].trim())) {
        rows.push(splitTableRow(lines[index]))
        index += 1
      }
      index -= 1
      blocks.push(<TableRenderer key={`table-${blocks.length}`} grid={[header, ...rows]} />)
      continue
    }
    const heading = line.match(/^#{1,3}\s+(.+)$/)
    if (heading) {
      flushParagraph(); flushBullets()
      blocks.push(<h3 key={`h-${blocks.length}`}>{renderInline(heading[1])}</h3>)
      continue
    }
    const bullet = line.match(/^(?:[-*•]|\d+\.)\s+(.+)$/)
    if (bullet) { flushParagraph(); bullets.push(bullet[1]); continue }
    flushBullets()
    paragraph.push(line)
  }
  flushParagraph(); flushBullets()
  return <div className="answer-text">{blocks.length ? blocks : <p>{renderInline(cleanText(text))}</p>}</div>
}

function splitTableRow(row: string) {
  return row.trim().replace(/^\||\|$/g, '').split('|').map(cell => cell.trim())
}

function renderInline(value: string): ReactNode {
  const normalized = value.replace(/\*\*(.+?)\*\*/g, '§B§$1§/B§').replace(/__(.+?)__/g, '§B§$1§/B§').replace(/\*([^*]+)\*/g, '§I§$1§/I§').replace(/_([^_]+)_/g, '§I§$1§/I§')
  const parts = normalized.split(/(§B§|§\/B§|§I§|§\/I§)/)
  let bold = false
  let italic = false
  return parts.map((part, index) => {
    if (part === '§B§') { bold = true; return null }
    if (part === '§/B§') { bold = false; return null }
    if (part === '§I§') { italic = true; return null }
    if (part === '§/I§') { italic = false; return null }
    const content = part.replace(/[*_]/g, '')
    return <Fragment key={index}>{bold ? <strong>{content}</strong> : italic ? <em>{content}</em> : content}</Fragment>
  })
}

export function CitationChip({doc,page,modality,onClick}:{doc:string;page:number;modality:Modality;onClick?:()=>void}) {
  return <button className="citation" onClick={onClick} aria-label={`Open ${modality} evidence from ${doc} page ${page} ${modality}`}>{doc} · p.{page}</button>
}

export function TableRenderer({grid}:{grid:string[][]}) {
  if (!grid.length) return <p>No table cells extracted.</p>
  return <div className="table-wrap"><table><thead><tr>{grid[0].map((c,i)=><th key={i}>{cleanText(c)}</th>)}</tr></thead><tbody>{grid.slice(1).map((row,i)=><tr key={i}>{row.map((c,j)=><td key={j}>{cleanText(c)}</td>)}</tr>)}</tbody></table></div>
}

export function EvidencePanel({items,onOpen}:{items:Evidence[];onOpen:(e:Evidence)=>void}) {
  const [tab,setTab]=useState<'all'|Modality>('all')
  const shown=items.filter(e=>tab==='all'||e.chunk.modality===tab)
  return <section className="evidence-section"><div className="section-heading"><div><div className="kicker">SOURCE EVIDENCE</div><h3>Why this answer is supported</h3></div><span className="result-count">{shown.length} source{shown.length===1?'':'s'}</span></div><div className="tabs" role="tablist" aria-label="Evidence modality">{(['all','text','table','figure'] as const).map(x=><button role="tab" aria-selected={tab===x} onClick={()=>setTab(x)} key={x}>{x}</button>)}</div><div className="evidence-grid" aria-hidden="true">{shown.map(e=><article id={`evidence-${e.chunk.chunk_id}`} role="button" aria-label={`Open ${e.chunk.modality} evidence from ${e.chunk.doc_id} page ${e.chunk.page}`} className={`evidence ${e.chunk.modality}`} key={e.chunk.chunk_id} onClick={()=>onOpen(e)} onKeyDown={event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();onOpen(e)}}} tabIndex={0}><div className="eyebrow">{e.chunk.doc_id} · PAGE {e.chunk.page} · {e.chunk.modality}</div>{e.chunk.modality==='table'&&e.chunk.raw_grid?<TableRenderer grid={e.chunk.raw_grid}/>:<>{e.chunk.modality==='figure'&&e.chunk.image_path&&<img className="figure-preview" src={`/api/figure/${e.chunk.chunk_id}`} alt={e.chunk.caption||`Figure on page ${e.chunk.page}`}/>}<p className="evidence-copy">{cleanText(e.chunk.caption||e.chunk.content||'Image-only page evidence')}</p></>}<small>Open the source page to inspect this evidence</small></article>)}</div></section>
}
