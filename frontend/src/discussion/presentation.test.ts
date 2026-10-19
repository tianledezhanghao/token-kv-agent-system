import { describe, expect, it } from 'vitest'
import {
  discussionStage,
  modeLabel,
  receptionForTurn,
  receiptLabel,
} from './presentation'
import type { DiscussionSession, Handoff, Turn } from './types'
const a = { sequence: 1, agent: 'A', round: 1, message_id: 'msg-a' } as Turn,
  b = { sequence: 2, agent: 'B', round: 1, message_id: 'msg-b', received_message_id: 'msg-a' } as Turn
const h = {
  sequence: 1,
  message_id: 'msg-a',
  source: 'A',
  target: 'B',
  discussion_round: 1,
  requested_mode: 'kv',
  requested_kv_strategy: 'rope_aligned_delta',
  executed_mode: 'token',
  executed_kv_strategy: null,
  append_succeeded: true,
  error: null,
  fallback_reason: null,
} as Handoff
const session = (handoffs: Handoff[] = [h]) =>
  ({
    turns: [a, b],
    handoffs,
    status: 'completed',
    phase: 'DISCUSSION_COMPLETED',
    config: { mode: 'mixed' },
  }) as DiscussionSession
describe('Receipt-grounded provenance', () => {
  it('labels only the first A turn as initial, without inventing a handoff', () => {
    expect(receptionForTurn(session(), a)).toEqual({ label: '初始发言' })
  })
  it('uses actual receipt rather than Mixed config or requested KV', () => {
    const r = receptionForTurn(session(), b)
    expect(r.label).toBe('接收 TOKEN')
    expect(r.handoff).toBe(h)
  })
  it('does not equate a request with actual execution', () => {
    expect(
      receptionForTurn(session([{ ...h, executed_mode: null }]), b).label,
    ).toBe('交接结果待确认')
  })
  it('never reports MIXED as a concrete received mode', () => {
    expect(receiptLabel({ ...h, executed_mode: 'mixed' })).toBe(
      '交接模式未确认（MIXED）',
    )
  })
  it('marks missing evidence as unknown', () => {
    expect(receptionForTurn(session([]), b)).toEqual({
      label: '接收来源待确认',
    })
  })
  it.each([
    { ...h, target: 'A' },
    { ...h, message_id: 'unrelated' },
    { ...h, source: 'B' },
    { ...h, discussion_round: 2 },
  ])('rejects an unrelated receipt %j', (receipt) => {
    expect(receptionForTurn(session([receipt]), b).label).toBe('接收来源待确认')
  })
  it('rejects ambiguous duplicate receipts', () => {
    expect(receptionForTurn(session([h, h]), b).label).toBe('接收来源待确认')
  })
  it('matches the second-round A turn to B→A of the first round', () => {
    const turn = { sequence: 3, agent: 'A', round: 2, message_id: 'msg-a2', received_message_id: 'msg-b' } as Turn
    const s = {
      ...session([
        {
          ...h,
          sequence: 2,
          message_id: 'msg-b',
          source: 'B',
          target: 'A',
          executed_mode: 'kv',
          executed_kv_strategy: 'rope_aligned_delta',
        },
      ]),
      turns: [a, b, turn],
    }
    expect(receptionForTurn(s, turn).label).toBe('接收 KV V1')
  })
  it('preserves failure even if an executed mode was recorded', () => {
    expect(
      receiptLabel({ ...h, error: 'append failed', append_succeeded: false }),
    ).toBe('交接失败 · TOKEN')
  })
  it('shows a successful fallback explicitly', () => {
    expect(receiptLabel({ ...h, fallback_reason: 'KV incompatible' })).toBe(
      '回退接收 TOKEN',
    )
  })
  it('does not show a failed fallback as received', () => {
    expect(
      receiptLabel({
        ...h,
        fallback_reason: 'KV incompatible',
        append_succeeded: false,
      }),
    ).toBe('回退未交付 · TOKEN')
  })
  it.each([
    ['token', null, 'TOKEN'],
    ['kv', 'raw_delta', 'KV V0'],
    ['kv', 'rope_aligned_delta', 'KV V1'],
    ['kv', 'v2', 'KV V2'],
    ['kv:v3', null, 'KV V3'],
    ['kv', null, 'KV（版本未确认）'],
    ['kv', 'future_repair', 'KV · future_repair'],
    ['mixed', null, 'MIXED'],
  ])('formats %s / %s without guessing', (mode, strategy, label) => {
    expect(modeLabel(mode, strategy)).toBe(label)
  })
  it('shows pending current handoff separately from the prior successful receipt', () => {
    const stage = discussionStage({
      ...session(),
      status: 'running',
      phase: 'HANDOFF_STARTED',
      agent: 'B',
      sequence: 2,
    })
    expect(stage.mode).toContain('本次交接 · 模式待确认')
    expect(stage.source).toBe('B')
    expect(stage.target).toBe('A')
    expect(stage.handoff).toBeUndefined()
  })
  it('does not hardcode TOKEN when a KV receipt exists', () => {
    const stage = discussionStage(
      session([
        {
          ...h,
          executed_mode: 'kv',
          executed_kv_strategy: 'v3',
          kv_repair_method: 'reported_repair',
        },
      ]),
    )
    expect(stage.mode).toContain('KV V3')
    expect(stage.repair).toBe('reported_repair')
  })
})

 it('labels actual suffix recompute receipts V2', () => {
   expect(modeLabel('kv', 'suffix_recompute_delta')).toBe('KV V2')
 })

 it('labels actual numeric slot strategy as V3', () => {
   expect(modeLabel('kv', 'numeric_slot_delta')).toBe('KV V3')
 })

describe('Distributed Receiver receipt trust boundary', () => {
  const receipt = { schema: 'remote_agent_commit_v1', architecture: 'distributed', session_id: 'd1', message_id: 'msg-a', handoff: 1, source: 'A', target: 'B', mode: 'token', kv_strategy: null, status: 'committed', cache_imported: true, state_committed: true, packet_sha256: 'abc', receiver_boot_id: 'boot-b' }
  const distributed = { ...h, architecture: 'distributed', session_id: 'd1', actual_execution_status: 'receiver_committed', receiver_receipt: receipt, delivery_identity: { session_id: 'd1', message_id: 'msg-a', handoff: 1, source: 'A', target: 'B', packet_sha256: 'abc', receiver_boot_id: 'boot-b' } } as Handoff
  it('accepts only a matching remote cache commit', () => expect(receiptLabel(distributed)).toBe('接收 TOKEN'))
  it.each(['session_id','message_id','source','target','receiver_boot_id','packet_sha256'])('rejects mismatched %s', key => {
    expect(receiptLabel({ ...distributed, receiver_receipt: { ...receipt, [key]: 'other' } })).toContain('提交未确认')
  })
  it.each(['cache_imported','state_committed'])('requires %s', key => {
    expect(receiptLabel({ ...distributed, receiver_receipt: { ...receipt, [key]: false } })).toContain('提交未确认')
  })
  it('never trusts legacy append flags without the distributed receipt', () => expect(receiptLabel({ ...distributed, receiver_receipt: null })).toContain('未记录有效回执'))
  it.each([['delivery_unknown','提交结果未知'],['receiver_rejected','明确拒绝'],['not_sent','传输失败']])('distinguishes %s', (actual_execution_status,label) => {
    expect(receiptLabel({ ...distributed, append_succeeded: false, actual_execution_status, receiver_receipt: null })).toContain(label)
  })
})
