export type Modality = 'text' | 'table' | 'figure'
export type PageName = 'search' | 'library' | 'evaluations' | 'history' | 'settings'
export interface BoundingBox { x0:number; y0:number; x1:number; y1:number }
export interface Chunk { chunk_id:string; doc_id:string; page:number; modality:Modality; content:string; caption:string; raw_grid?:string[][]; image_path?:string; bbox?:BoundingBox; metadata?:Record<string,unknown> }
export interface Evidence { chunk:Chunk; dense_score?:number; bm25_score?:number; fused_score:number; explanation?:Record<string, unknown> }
export interface Answer { text:string; citations:{doc_id:string;page:number;modality:Modality;chunk_id?:string}[]; evidence:Evidence[]; grounded:boolean; latency_ms:number; model_id:string; prompt_hash:string; provider?:string; metadata?:Record<string,unknown> }
export interface DocumentRecord { doc_id:string; filename?:string; company?:string; fiscal_year?:string; pages?:number; text_chunks?:number; tables?:number; figures?:number; ocr_pages?:number; warning_count?:number; warnings?:string[]; processing_state?:string; last_ingestion_time?:string; last_index_time?:string }
export interface DocumentDetail { document:DocumentRecord; tables:Chunk[]; figures:Chunk[] }
export interface JobRecord { job_id:string; filename:string; state:string; stage:string; progress:number; pages_processed:number; total_pages?:number; warning_count:number; error?:string }
export interface EvaluationRun { run_id:string; metrics:Record<string, unknown>; config?:Record<string,unknown>; environment?:Record<string,unknown>; git_sha?:string; per_question?:Record<string,unknown>[] }
export interface HistoryEntry { history_id:string; title:string; question:string; payload:Record<string,unknown>; pinned:boolean; created_at:string }
export interface RuntimeSettings { final_k:number; candidate_k:number; fusion:'rrf'|'weighted'; rrf_k:number; dense_weight:number; sparse_weight:number; provider:string; model:string; vision:boolean; demo_mode:boolean }
export interface PageMetadata { page:number; page_count:number; width:number; height:number; rotation:number }
