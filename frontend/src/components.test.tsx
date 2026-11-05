import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { CitationChip, EvidencePanel, ReadableAnswer, TableRenderer } from './components'

describe('evidence components',()=>{
 it('renders an accessible citation chip',()=>{render(<CitationChip doc="ACME" page={2} modality="table"/>);expect(screen.getByRole('button',{name:/ACME page 2 table/})).not.toBeNull()})
 it('renders table cells',()=>{render(<TableRenderer grid={[["Metric","FY24"],["Revenue","120"]]}/>);expect(screen.getByText('Revenue')).not.toBeNull()})
 it('renders markdown answer formatting without leaking syntax',()=>{render(<ReadableAnswer text={`**Revenue:** increased by *12%*

| Metric | FY24 |
| --- | --- |
| Sales | $120 |`}/>);expect(screen.getByText('Revenue:')).toHaveProperty('tagName','STRONG');expect(screen.getByText('12%')).toHaveProperty('tagName','EM');expect(screen.queryByText('**')).toBeNull();expect(screen.getByText('Sales')).not.toBeNull()})
 it('filters evidence tabs',()=>{render(<EvidencePanel onOpen={()=>{}} items={[{chunk:{chunk_id:'1',doc_id:'A',page:1,modality:'text',content:'alpha',caption:''},fused_score:1},{chunk:{chunk_id:'2',doc_id:'A',page:2,modality:'figure',content:'beta',caption:'chart'},fused_score:.5}]}/>);fireEvent.click(screen.getByRole('tab',{name:'figure'}));expect(screen.queryByText('alpha')).toBeNull();expect(screen.getByText('chart')).not.toBeNull()})
})
