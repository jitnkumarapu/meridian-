import AxeBuilder from '@axe-core/playwright'
import { expect, test } from '@playwright/test'

const answer = {
  text: 'Revenue was 120.',
  citations: [{ doc_id: 'acme', page: 2, modality: 'table', chunk_id: 't1' }],
  evidence: [{ chunk: { chunk_id: 't1', doc_id: 'acme', page: 2, modality: 'table', content: 'Revenue 120', caption: '', raw_grid: [['Metric', 'FY24'], ['Revenue', '120']], bbox: { x0: 10, y0: 20, x1: 100, y1: 80 } }, fused_score: 1 }],
  grounded: true, latency_ms: 4, model_id: 'mock', prompt_hash: '1234567890abcdef',
}
const settings = { final_k: 5, candidate_k: 20, fusion: 'rrf', rrf_k: 60, dense_weight: .6, sparse_weight: .4, provider: 'mock', model: 'mock', vision: false, demo_mode: true }
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=', 'base64')

test.beforeEach(async ({ page }) => {
  await page.route('**/api/**', async route => {
    const url = route.request().url()
    if (url.endsWith('.png') || url.includes('/figure/')) return route.fulfill({ status: 200, contentType: 'image/png', body: png })
    if (url.includes('/api/ask')) return route.fulfill({ json: answer })
    if (url.includes('/api/page/') && url.includes('/metadata')) return route.fulfill({ json: { page: 2, page_count: 3, width: 200, height: 400, rotation: 0 } })
    if (url.match(/\/api\/documents\/[^/]+$/)) return route.fulfill({ json: { document: { doc_id: 'acme', company: 'ACME', pages: 3, warning_count: 0 }, tables: [], figures: [] } })
    if (url.endsWith('/api/documents')) return route.fulfill({ json: [{ doc_id: 'acme', company: 'ACME', pages: 3, text_chunks: 2, tables: 1, figures: 1, ocr_pages: 0, warning_count: 0, processing_state: 'ready' }] })
    if (url.endsWith('/api/jobs')) return route.fulfill({ json: [] })
    if (url.match(/\/api\/evaluations\/[^/]+$/)) return route.fulfill({ json: { run_id: 'run-1', metrics: { systems: { multimodal: { mrr: { mean: 1 } } } }, per_question: [] } })
    if (url.endsWith('/api/evaluations')) return route.fulfill({ json: [{ run_id: 'run-1', metrics: {} }] })
    if (url.includes('/api/history')) return route.fulfill({ json: [{ history_id: 'h1', title: 'Revenue', question: 'Revenue?', payload: { answer: '120' }, pinned: false, created_at: '2026-01-01T00:00:00Z' }] })
    if (url.includes('/api/settings')) return route.fulfill({ json: settings })
    return route.fulfill({ json: {} })
  })
})

test('search, compare, evidence drawer, navigation, theme, and accessibility', async ({ page }) => {
  const errors: string[] = []
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()) })
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/')
  await page.getByLabel('Question').fill('Revenue?')
  await page.locator('form').getByRole('button', { name: 'Ask' }).click()
  await expect(page.getByText('Revenue was 120.')).toHaveCount(1)
  await page.getByRole('button', { name: /Open table evidence/ }).click()
  await expect(page.getByRole('dialog', { name: 'Source evidence' })).toBeVisible()
  await expect(page.getByTitle('Extracted evidence region')).toBeVisible()
  await page.getByRole('button', { name: 'Close' }).click()
  await page.getByRole('button', { name: /Library/ }).click()
  await expect(page.getByText('ACME')).toBeVisible()
  await page.getByLabel('Toggle theme').click()
  const accessibility = await new AxeBuilder({ page }).analyze()
  expect(accessibility.violations).toEqual([])
  expect(errors).toEqual([])
})

test('mobile layout has no document-level horizontal overflow', async ({ page }) => {
  await page.goto('/')
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth)
  expect(overflow).toBe(false)
  await expect(page.getByText('Not financial advice')).toBeVisible()
})
