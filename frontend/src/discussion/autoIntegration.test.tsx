import { Transmission } from './DiscussionDetails'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { defaultNetworkAuto, draftFromConfig, networkAutoConfig, parseProfile } from './autoConfig'
import { DecisionFailure, DecisionInsights, NetworkHistory } from './HandoffInsights'
import { deliveryForTurn } from './presentation'
import { projectDiscussion, restoreDiscussion } from './state'
import { startDiscussion } from './api'
import type { AutoDecision, DiscussionConfig, DiscussionSession, Handoff, Turn } from './types'
import type { DashboardEvent } from '../types/dashboard'
const identity = { sender: { model_id: 'fixture-model' }, receiver: { model_id: 'fixture-model' } }
const profile = { version: 'fixture-v1', provenance: 'synthetic/test_only', model_identity: identity, stages: { token_handoff: { fixed_ms: 100 }, kv_base: { fixed_ms: 1 }, kv_contract_startup: { fixed_ms: 0 } }, sample_count: 0, message_token_range: [1, 128], cache_token_range: [0, 4096], notes: 'Synthetic test fixture, never measurement.' }
const draft = { ...defaultNetworkAuto, autoPolicy: 'performance_experiment' as const, idealized: false, profileText: JSON.stringify(profile) }
const turn = { message_id: 'A-message-unique', agent: 'A', sequence: 7, round: 1 } as Turn
const receipt = { message_id: turn.message_id, source: 'A', target: 'B', sequence: 1, requested_mode: 'auto', selected_mode: 'kv', selected_kv_strategy: 'rope_aligned_delta', executed_mode: 'token', append_succeeded: true, error: null } as Handoff
const session = { session_id: 'new-session', revision: 1, created_at: '2026-10-14T10:00:00Z', handoffs: [], turns: [turn], auto_decisions: [], status: 'running', config: { mode: 'auto', network_mode: 'trace', bandwidth_trace: [20, 1000] } } as unknown as DiscussionSession
const decision: AutoDecision = { auto_policy: 'quality_first', selected_candidate: 'token', selection_reason: 'sole_safe_TOKEN_uncalibrated', profile_provenance: 'not_measured', quality_risk: false, candidate_estimates: [{ candidate: 'token', predicted_local_ms: null, simulated_transfer_ms: 2, predicted_total_ms: null, predicted_wire_bytes: 30, cost_status: 'uncalibrated' }], excluded_candidates: [{ candidate: 'kv:v0', reason: 'quality_not_certified' }, { candidate: 'kv:v1', reason: 'model_profile_mismatch' }, { candidate: 'kv:v2', reason: 'missing_startup_cost_profile' }, { candidate: 'kv:v3', reason: 'known_quality_regression' }] }
afterEach(() => { cleanup(); vi.unstubAllGlobals() })
describe('REST network and AUTO configuration', () => {
  it('new AUTO defaults use the fixed performance policy and explicit synthetic assumptions', () => {
    expect(networkAutoConfig(defaultNetworkAuto, 'auto')).toMatchObject({ auto_policy: 'performance_experiment', idealized_simulation: true, cost_profile: null, allow_experimental_v3: false, auto_candidates: ['token', 'kv:v0'] })
  })
  it('manual ignores inactive AUTO draft and clears trace', () => { expect(networkAutoConfig({ ...draft, networkMode: 'manual', trace: 'bad' }, 'kv:v3')).toEqual({ network_mode: 'manual', bandwidth_mbps: 20, base_latency_ms: 0, bandwidth_trace: [] }) })
  it('maps handoff trace and latency to strict REST numbers', () => { expect(networkAutoConfig({ ...defaultNetworkAuto, networkMode: 'trace', latency: '2.5' }, 'mixed')).toEqual({ network_mode: 'trace', bandwidth_mbps: 20, base_latency_ms: 2.5, bandwidth_trace: [20, 1000, 50, 500] }) })
  it.each(['', '20,,50', '0', '-1', 'Infinity', 'text'])('rejects invalid trace %s', trace => { expect(() => networkAutoConfig({ ...defaultNetworkAuto, networkMode: 'trace', trace }, 'auto')).toThrow(/轨迹/) })
  it.each(['', '0', '-1', 'Infinity'])('rejects invalid bandwidth %s', bandwidth => { expect(() => networkAutoConfig({ ...defaultNetworkAuto, networkMode: 'manual', bandwidth }, 'token')).toThrow(/带宽/) })
  it('rejects negative latency', () => { expect(() => networkAutoConfig({ ...defaultNetworkAuto, latency: '-1' }, 'token')).toThrow(/延迟/) })
  it('requires explicit performance profile and V3 permission', () => {
    expect(() => networkAutoConfig({ ...draft, profileText: '' }, 'auto')).toThrow(/Profile/)
    expect(() => networkAutoConfig({ ...draft, selectedKv: 'v3' }, 'auto')).toThrow(/V3/)
    expect(networkAutoConfig({ ...draft, selectedKv: 'v3', allowV3: true }, 'auto', identity)).toMatchObject({ auto_candidates: ['token', 'kv:v3'], allow_experimental_v3: true, cost_profile: profile })
  })
  it('idealized source is explicit and disjoint from imported profiles', () => { expect(networkAutoConfig({ ...draft, idealized: true }, 'auto')).toMatchObject({ idealized_simulation: true, cost_profile: null, auto_candidates: ['token','kv:v0'] }) })
  it('restores full AUTO config as a draft without inventing a default profile', () => { const config = networkAutoConfig(draft, 'auto', identity) as DiscussionConfig; expect(draftFromConfig(config)).toEqual({ ...draft, synthetic: { a: '20', b: '2', c: '0.002', d: '5', e: '0.05', f: '0.2' }, profileText: JSON.stringify(profile, null, 2) }) })
  it('POST sends inline profile JSON and network values without a file path', async () => {
    const fetch = vi.fn(async (_path: string, _init?: RequestInit) => ({ ok: true, json: async () => session })); vi.stubGlobal('fetch', fetch)
    const c = { question: 'Q', rounds: 1, mode: 'auto', max_new_tokens: 16, summarizer: 'A', handoff_plan: [], ...networkAutoConfig(draft, 'auto', identity) } as DiscussionConfig
    await startDiscussion(c, 'accepted-session')
    const body = JSON.parse(String(fetch.mock.calls[0][1]?.body))
    expect(body).toEqual({ ...c, session_id: 'accepted-session' }); expect(body.cost_profile).toEqual(profile); expect(body).not.toHaveProperty('cost_profile_path')
  })
})
describe('incremental profile validation', () => {
  it('validates provenance and current backend identity', () => { expect(parseProfile(JSON.stringify(profile), identity)?.provenance).toBe('synthetic/test_only'); expect(() => parseProfile(JSON.stringify(profile), { sender: { model_id: 'other' } })).toThrow(/身份/) })
  it.each([{ ...profile, provenance: 'measured' }, { ...profile, provenance: 'measured_incremental' }, { ...profile, semantics: 'canonical' }, { ...profile, notes: '' }, { ...profile, cache_token_range: [100, 1] }, { ...profile, stages: { token_handoff: { fixed_ms: -1 } } }, { ...profile, cost_profile_path: '/server/file' }])('rejects incompatible profile %j', p => { expect(() => parseProfile(JSON.stringify(p))).toThrow() })
  it('rejects malformed JSON and paths', () => { expect(() => parseProfile('{')).toThrow(/JSON/); expect(() => parseProfile('"/server/file"')).toThrow(/字段/) })
})
describe('decision and receipt presentation', () => {
  it('shows all candidates, quality reasons and null calibration without zeros', () => {
    render(<DecisionInsights decision={decision} />)
    const row = within(screen.getByRole('row', { name: /TOKEN/ }))
    expect(row.getAllByText('未校准')).toHaveLength(2)
    for (const text of ['质量未认证', '模型身份不匹配', '启动成本', '已知质量退化']) expect(screen.getByText(new RegExp(text))).toBeInTheDocument()
    expect(screen.queryByText('0.00 ms')).not.toBeInTheDocument()
  })
  it('permanently labels synthetic predictions test only', () => { render(<DecisionInsights decision={{ ...decision, profile_provenance: 'synthetic/test_only', auto_policy: 'performance_experiment', quality_risk: true }} />); expect(screen.getByText(/模拟成本 · 测试/)).toBeInTheDocument(); expect(screen.getByText(/性能实验：KV/)).toBeInTheDocument() })
  it('uses message ID despite unrelated sequence and selected KV', () => { expect(deliveryForTurn({ ...session, handoffs: [receipt] }, turn).label).toBe('B 已接收 TOKEN'); expect(deliveryForTurn({ ...session, handoffs: [{ ...receipt, message_id: 'other' }] }, turn).handoff).toBeUndefined() })
  it.each(['raw_delta', 'rope_aligned_delta', 'suffix_recompute_delta', 'numeric_slot_delta'])('labels concrete receipt strategy %s', strategy => { expect(deliveryForTurn({ ...session, handoffs: [{ ...receipt, executed_mode: 'kv', executed_kv_strategy: strategy }] }, turn).label).toMatch(/已接收 KV V[0-3]/) })
  it('does not convert selection, generation or failed receipt into delivery', () => {
    expect(deliveryForTurn({ ...session, active_handoff: { sequence: 1, message_id: turn.message_id, agent: 'A', target: 'B', requested_mode: 'auto', selected_mode: 'kv', selected_kv_strategy: 'rope_aligned_delta' } }, turn).label).toContain('尚未交付')
    expect(deliveryForTurn({ ...session, handoffs: [{ ...receipt, append_succeeded: false, error: 'failed' }] }, turn).label).toContain('失败')
    expect(deliveryForTurn({ ...session, handoffs: [receipt, receipt] }, turn).label).toContain('歧义')
  })
  it('shows decision failure and structured exclusions from backend error', () => { render(<DecisionFailure record={{ sequence: 1, error: 'no eligible AUTO candidates: [{"candidate":"kv:v3","reason":"known_quality_regression"}]' }} />); expect(screen.getByText('KV V3：已知质量退化')).toBeInTheDocument() })
  it('uses snapshot progress including exhaustion and next hold_last', () => { render(<NetworkHistory session={{ ...session, handoffs: [{ ...receipt, network: { bandwidth_mbps: 1000, base_latency_ms: 2, source: 'rest_session_config', network_mode: 'trace', handoff_index: 3, simulation_step: 2, trace_exhausted: true } }] }} />); expect(screen.getByText(/横轴是 Handoff 序号/)).toBeInTheDocument(); expect(screen.getByText('Handoff 1')).toBeInTheDocument(); expect(screen.getByText('已耗尽，保持末项')).toBeInTheDocument() })
})
describe('WS/REST consistent AUTO projections', () => {
  const event = (revision: number, type: string, metadata = {}) => ({ type, timestamp: '2026-10-14T10:01:00Z', metadata: { session_id: session.session_id, revision, status: 'running', sequence: 1, agent: 'A', target: 'B', ...metadata } }) as unknown as DashboardEvent
  it('retains a decision before delivery, deduplicates replay and repairs out-of-order gap', () => {
    const selected = projectDiscussion(session, event(2, 'AUTO_DECISION', { message_id: turn.message_id, decision }))!
    expect(selected.auto_decisions).toHaveLength(1); expect(selected.handoffs).toHaveLength(0); expect(selected.active_handoff?.decision?.selected_candidate).toBe('token')
    expect(projectDiscussion(selected, event(2, 'AUTO_DECISION', { decision }))).toBe(selected)
    expect(projectDiscussion(selected, event(4, 'HANDOFF_COMPLETED', { handoff: receipt }))).toBe(selected)
    const started = projectDiscussion(selected, event(3, 'HANDOFF_STARTED', { message_id: turn.message_id, auto_decision: decision }))!
    const done = projectDiscussion(started, event(4, 'HANDOFF_COMPLETED', { handoff: receipt }))!
    expect(done.handoffs).toHaveLength(1); expect(done.auto_decisions).toHaveLength(1); expect(done.active_handoff).toBeNull()
    expect(restoreDiscussion(session, done)).toBe(done)
  })
  it('retains failed decisions through terminal and never polls old events into a new session', () => {
    const failed = projectDiscussion(session, event(2, 'AUTO_DECISION_FAILED', { error: 'no eligible', decision_status: 'failed', message_id: turn.message_id }))!
    const ended = projectDiscussion(failed, event(3, 'DISCUSSION_FAILED', { status: 'failed' }))!
    expect(ended.auto_decisions?.[0].error).toBe('no eligible'); expect(ended.active_handoff).toBeNull(); expect(ended.handoffs).toHaveLength(0)
    expect(projectDiscussion(ended, { ...event(4, 'AUTO_DECISION'), metadata: { session_id: 'old', revision: 4 } })).toBe(ended)
  })
})


it('does not claim a quality filter for a historical uncalibrated performance policy', () => {
  render(<DecisionInsights decision={{ ...decision, auto_policy: 'performance_experiment', quality_risk: true, excluded_candidates: [{candidate:'kv:v0',reason:'missing_cost_profile'}] }} />)
  expect(screen.getByText('选择 TOKEN：这是唯一可用候选，本地成本未校准。')).toBeInTheDocument()
  expect(screen.queryByText(/当前仅 TOKEN 获得默认质量许可/)).not.toBeInTheDocument()
})

it('groups all handoff diagnostics and preserves original enums in collapsed advanced data', () => {
  const handoff = { ...receipt, executed_mode: 'kv', executed_kv_strategy: 'rope_aligned_delta', kv_repair_method: 'post_rope_rephase', auto_decision: { ...decision, profile_provenance: 'synthetic/test_only' } }
  render(<Transmission handoff={handoff} open />)
  for (const name of ['通信决策', '成本分析', '网络环境', 'Cache 与 KV 修正', '交付内容']) expect(screen.getByRole('region', { name })).toBeInTheDocument()
  expect(screen.getByText('模拟成本 · 测试 · 仅用于路由测试，预测不代表实测性能')).toBeInTheDocument()
  expect(screen.getByText('RoPE 位置修正')).toBeInTheDocument()
  const advanced = screen.getByText('高级诊断 · 完整交接元数据').closest('details')!
  expect(advanced).not.toHaveAttribute('open')
  expect(advanced.querySelector('pre')).toHaveTextContent('post_rope_rephase')
  expect(advanced.querySelector('pre')).toHaveTextContent('rope_aligned_delta')
  expect(advanced.querySelector('pre')).toHaveTextContent('synthetic/test_only')
  fireEvent.click(within(advanced).getByText('高级诊断 · 完整交接元数据'))
  expect(JSON.parse(advanced.querySelector('pre')!.textContent!)).toEqual(handoff)
})

it('projects network updates without rewriting frozen configuration or active decisions', () => {
  const before = { config: { mode: 'trace', bandwidth_mbps: 20, base_latency_ms: 0, trace_mbps: [20, 1000], source: 'initial' }, config_revision: 0, next_unfrozen_handoff_index: 3 }
  const after = { ...before, config: { ...before.config, mode: 'manual', bandwidth_mbps: 50, trace_mbps: [] }, config_revision: 1 }
  const active = { sequence: 2, network: { bandwidth_mbps: 1000, base_latency_ms: 0, source: 'initial', config_revision: 0 } }
  const base = { ...session, phase: 'HANDOFF_STARTED', active_handoff: active } as unknown as DiscussionSession
  const ev = { type: 'DISCUSSION_NETWORK_UPDATED', timestamp: '2026-10-15T00:00:00Z', metadata: { session_id: base.session_id, revision: 2, status: 'running', phase: 'HANDOFF_STARTED', sequence: 2, agent: 'B', round: 1, network: after, network_update: { before, after, config_revision: 1 } } } as unknown as DashboardEvent
  const result = projectDiscussion(base, ev)!
  expect(result.network).toEqual(after)
  expect(result.phase).toBe('HANDOFF_STARTED')
  expect(result.active_handoff).toBe(active)
  expect(result.config).toBe(base.config)
  expect(result.network_updates).toHaveLength(1)
  expect(projectDiscussion(result, ev)).toBe(result)
  expect(restoreDiscussion(result, base)).toBe(result)
})

it('renders only the selected pair and labels predicted measured calibration separately from observations', () => {
  render(<DecisionInsights decision={{ ...decision, profile_provenance: 'measured', candidate_set: ['token','kv:v2'], selected_kv: 'kv:v2', selection_reason: 'uncertainty_overlap_prefer_token', candidate_estimates: [{ candidate: 'token', predicted_local_ms: 100, predicted_wire_bytes: 42, simulated_transfer_ms: 1, predicted_total_ms: 101, uncertainty_ms: 25, cost_source: 'predicted', cost_status: 'estimated' }, { candidate: 'kv:v2', predicted_local_ms: 90, predicted_wire_bytes: 10000, simulated_transfer_ms: 2, predicted_total_ms: 92, uncertainty_ms: 23, cost_source: 'predicted', cost_status: 'estimated' }], excluded_candidates: [] }} />)
  expect(screen.getByText(/不确定性区间重叠/)).toBeInTheDocument()
  expect(screen.getByText(/实测校准 → 有界预测/)).toBeInTheDocument()
  expect(screen.queryByRole('row', { name: /KV V0/ })).not.toBeInTheDocument()
  expect(screen.getByRole('row', { name: /KV V2/ })).toBeInTheDocument()
  expect(screen.getByText('25.00 ms')).toBeInTheDocument()
})

it('accepts the new schema without requiring exact message SHA, with identity validation', () => {
  const generalized = { schema: 'generalized_incremental_cost_v1', version: 'fixture', provenance: 'measured', units: 'ms', model_identity: identity, environment: {}, points: [{ candidate: 'token' }], sample_count: 2, evidence_sha256: 'a'.repeat(64) }
  expect(parseProfile(JSON.stringify(generalized), identity)?.schema).toBe('generalized_incremental_cost_v1')
  expect(() => parseProfile(JSON.stringify({ ...generalized, provenance: 'synthetic' }))).toThrow(/实测/)
  expect(() => parseProfile(JSON.stringify(generalized), { sender: { model_id: 'different' } })).toThrow(/身份/)
})


it('normalizes legacy quality drafts only at new-experiment submission', () => {
  const historical = {mode:'auto', auto_policy:'quality_first', idealized_simulation:false} as DiscussionConfig
  const restored = draftFromConfig(historical)
  expect(restored.autoPolicy).toBe('quality_first')
  expect(restored.idealized).toBe(false)
  expect(networkAutoConfig({...restored, idealized:true}, 'auto')).toMatchObject({auto_policy:'performance_experiment', idealized_simulation:true})
  expect(historical).toEqual({mode:'auto',auto_policy:'quality_first',idealized_simulation:false})
})


it('retains the measured Profile path after disabling idealized simulation', () => {
  const measured = {...profile, provenance:'measured_incremental', sample_count:3, notes:'Measured fixture for configuration validation.'}
  expect(networkAutoConfig({...defaultNetworkAuto, idealized:false, profileText:JSON.stringify(measured)}, 'auto', identity)).toMatchObject({auto_policy:'performance_experiment', idealized_simulation:false, cost_profile:measured, synthetic_assumptions:null})
})
