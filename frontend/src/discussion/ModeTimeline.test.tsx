import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { handoffDescription, ModeTimeline, timelineHandoffs } from './ModeTimeline'
import { projectDiscussion } from './state'
import type { DiscussionSession, Handoff } from './types'

const h = (sequence: number, executed_mode = 'token', extra: Partial<Handoff> = {}) => ({ sequence, executed_mode, source: sequence % 2 ? 'A' : 'B', target: sequence % 2 ? 'B' : 'A', append_succeeded: true, error: null, ...extra } as Handoff)
const s = (handoffs: Handoff[] = []) => ({ session_id: 'one', revision: 1, handoffs, turns: [] } as unknown as DiscussionSession)
afterEach(cleanup)

describe('receipt-backed mode timeline', () => {
  it('draws TOKEN→KV→TOKEN steps and distinguishes sender shapes with the latest ring', () => {
    const { container } = render(<ModeTimeline session={s([h(3), h(1), h(2, 'kv', { executed_kv_strategy: 'raw_delta' })])} />)
    expect(container.querySelector('.timeline-step')).toHaveAttribute('d', 'M48 94 H313V32 H578V94')
    const nodes = container.querySelectorAll('.timeline-node')
    expect(nodes).toHaveLength(3)
    expect(nodes[0].querySelector('circle.timeline-marker')).not.toBeNull()
    expect(nodes[1].querySelector('path.timeline-marker')).not.toBeNull()
    expect(nodes[2]).toHaveClass('latest')
    fireEvent.mouseEnter(nodes[1])
    expect(screen.getByRole('tooltip')).toHaveTextContent('KV V0')
    expect(screen.getByRole('tooltip')).toHaveTextContent('Agent B → Agent A')
    expect(fireEvent.keyDown(nodes[1], { key: 'Escape' })).toBe(false)
    expect(screen.queryByRole('tooltip')).toBeNull()
  })
  it('spaces sparse historical records without overlapping adjacent nodes', () => {
    const { container } = render(<ModeTimeline session={s([h(1), h(2), h(40)])} mini />)
    expect(container.querySelector('.timeline-step')).toHaveAttribute('d', 'M48 36 H163V36 H278V36')
  })
  it('keeps consecutive modes on the same track, including a single point', () => {
    const { container, rerender } = render(<ModeTimeline session={s([h(1, 'kv'), h(2, 'kv')])} />)
    expect(container.querySelector('.timeline-step')).toHaveAttribute('d', 'M48 32 H578V32')
    rerender(<ModeTimeline session={s([h(1)])} mini />)
    expect(container.querySelectorAll('.timeline-node')).toHaveLength(1)
    expect(container.querySelector('.timeline-step')).toHaveAttribute('d', 'M48 36')
  })
  it('does not invent successful deliveries from decisions, requests, failed or unknown records', () => {
    expect(timelineHandoffs([h(1, 'auto'), h(2, 'kv', { append_succeeded: false }), h(3, 'token', { error: 'failed' }), h(4, 'kv', { append_succeeded: null }), h(5, 'token', { architecture: 'distributed' }), h(6, 'token', { sequence: NaN })])).toEqual([])
    render(<ModeTimeline session={s()} mini />)
    expect(screen.getByText('暂无已确认交接')).toBeInTheDocument()
  })
  it('handles historical missing metadata without claiming a KV version or filling zero costs', () => {
    const old = h(1, 'kv')
    expect(handoffDescription(old)).toContain('版本未确认')
    expect(handoffDescription(old)).not.toContain('成本')
    render(<ModeTimeline session={s([old])} />)
    fireEvent.focus(screen.getByRole('button', { name: /交接 #1/ }))
    expect(screen.getByRole('tooltip')).toHaveTextContent('版本未确认')
  })
  it('shows finite costs and network snapshots, keeps zero and hides nonfinite values', () => {
    const description = handoffDescription(h(1, 'kv:v3', { predicted_local_ms: 0, predicted_total_ms: NaN, observed_local_ms: 2, auto_decision: { selected_candidate: 'kv:v3', candidate_estimates: [{ candidate: 'kv:v3', simulated_transfer_ms: 3 }] } as Handoff['auto_decision'], network: { channel: { snr_db: 4, capacity_mbps: 5, effective_rate_mbps: 6 } } as Handoff['network'] }))
    expect(description).toContain('KV V3')
    expect(description).toContain('预测本地成本：0.00 ms')
    expect(description).toContain('预测模拟网络成本：3.00 ms')
    expect(description).toContain('SNR：4.00 dB')
    expect(description).not.toContain('NaN')
  })
  it('limits mini to eight while full retains all forty handoffs without changing session data', () => {
    const session = s(Array.from({ length: 40 }, (_, i) => h(i + 1)))
    const { container } = render(<><ModeTimeline session={session} mini /><ModeTimeline session={session} /></>)
    expect(within(screen.getByRole('region', { name: '最近通信模式切换' })).getAllByRole('button')).toHaveLength(8)
    expect(container.querySelectorAll('.mode-timeline-full .timeline-node')).toHaveLength(40)
    expect(session.handoffs[0].sequence).toBe(1)
  })
  it('adds exactly one point after HANDOFF_COMPLETED and replays snapshots without extra nodes', () => {
    const original = s([h(1)])
    const { container, rerender } = render(<ModeTimeline session={original} mini />)
    const pending = projectDiscussion(original, { source: null, target: null, mode: null, payload_bytes: null, transmission_latency_ms: null, estimated_e2e_latency_ms: null, type: 'HANDOFF_STARTED', timestamp: '2026-10-19', metadata: { session_id: 'one', revision: 2, sequence: 2 } } as Parameters<typeof projectDiscussion>[1])!
    rerender(<ModeTimeline session={pending} mini />)
    expect(container.querySelectorAll('.timeline-node')).toHaveLength(1)
    const complete = projectDiscussion(pending, { source: null, target: null, mode: null, payload_bytes: null, transmission_latency_ms: null, estimated_e2e_latency_ms: null, type: 'HANDOFF_COMPLETED', timestamp: '2026-10-19', metadata: { session_id: 'one', revision: 3, handoff: h(2, 'kv:v2') } } as Parameters<typeof projectDiscussion>[1])!
    rerender(<ModeTimeline session={complete} mini />)
    expect(container.querySelectorAll('.timeline-node')).toHaveLength(2)
    rerender(<ModeTimeline session={JSON.parse(JSON.stringify(complete))} mini />)
    expect(container.querySelectorAll('.timeline-node')).toHaveLength(2)
    fireEvent.click(screen.getByRole('button', { name: /交接 #2/ }))
    rerender(<ModeTimeline session={{ ...s([h(1)]), session_id: 'other' }} mini />)
    expect(screen.queryByRole('tooltip')).toBeNull()
  })
})
