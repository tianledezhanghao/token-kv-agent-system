import type { DiscussionSession, Handoff, Turn } from './types'

export const phaseNames: Record<string, string> = {
  initializing: '初始化上下文',
  DISCUSSION_WORKER_READY: 'Worker 上下文已准备',
  DISCUSSION_STARTED: '初始化上下文',
  AGENT_TURN_STARTED: '模型生成中',
  AGENT_TURN_COMPLETED: '发言已生成',
  AUTO_DECISION: 'AUTO 已决策 · 尚未交付',
  AUTO_DECISION_FAILED: 'AUTO 决策失败',
  HANDOFF_STARTED: '通信进行中 · 尚未确认交付',
  HANDOFF_COMPLETED: '通信已完成',
  HANDOFF_UNCONFIRMED: '提交结果未知 · 需对账',
  HANDOFF_FAILED: '通信失败',
  DISCUSSION_FINALIZING: '最终总结生成中',
  DISCUSSION_COMPLETED: '讨论完成',
  DISCUSSION_FAILED: '讨论失败',
}
export const manualModes = [
  { value: 'token', label: 'TOKEN · 增量 Prefill' },
  { value: 'kv:v0', label: 'KV V0 · Raw Delta（实验性）' },
  { value: 'kv:v1', label: 'KV V1 · RoPE aligned（实验性）' },
  { value: 'kv:v2', label: 'KV V2 · 连续尾段重算（实验性）' },
  { value: 'kv:v3', label: 'KV V3 · 数字槽位修复（实验性）' },
] as const

// Never infer a KV version from the mere presence of a KV packet.
export function modeLabel(
  mode: string | null | undefined,
  strategy?: string | null,
): string {
  if (!mode) return '未确认'
  const m = mode.toLowerCase()
  if (m === 'token') return 'TOKEN'
  if (m === 'auto') return 'AUTO'
  if (m === 'mixed') return 'MIXED'
  if (/^kv:v\d+$/.test(m)) return `KV ${m.slice(3).toUpperCase()}`
  if (m === 'kv') {
    if (strategy === 'raw_delta') return 'KV V0'
    if (strategy === 'rope_aligned_delta') return 'KV V1'
    if (strategy === 'suffix_recompute_delta') return 'KV V2'
    if (strategy === 'numeric_slot_delta') return 'KV V3'
    if (strategy && /^(kv:)?v\d+$/i.test(strategy))
      return `KV ${strategy.replace(/^kv:/i, '').toUpperCase()}`
    return strategy ? `KV · ${strategy}` : 'KV（版本未确认）'
  }
  return mode
}
export function receiverCommitted(h: Handoff): boolean {
  if (h.architecture !== 'distributed') return h.append_succeeded === true
  const r = h.receiver_receipt
  return !!r && h.append_succeeded === true && r.schema === 'remote_agent_commit_v1' && r.status === 'committed' && r.state_committed === true && r.cache_imported === true && r.session_id === h.session_id && r.message_id === h.message_id && r.handoff === h.sequence && r.source === h.source && r.target === h.target && r.mode === h.executed_mode && r.kv_strategy === h.executed_kv_strategy && !!h.delivery_identity && typeof h.delivery_identity.packet_sha256 === 'string' && typeof h.delivery_identity.receiver_boot_id === 'string' && Object.entries(h.delivery_identity).every(([key,value]) => (r as unknown as Record<string, unknown>)[key] === value)
}
export function receiptLabel(h: Handoff): string {
  if (h.architecture === 'distributed') {
    if (h.actual_execution_status === 'delivery_unknown') return '提交结果未知 · 需对账'
    if (h.actual_execution_status === 'receiver_rejected') return 'Receiver 明确拒绝'
    if (h.actual_execution_status === 'not_sent') return '传输失败 · 未发送'
    if (!receiverCommitted(h)) return 'Receiver 提交未确认 · 未记录有效回执'
  }
  const actual = modeLabel(h.executed_mode, h.executed_kv_strategy)
  if (h.error || h.append_succeeded === false)
    return h.fallback_reason ? `回退未交付 · ${actual}` : `交接失败 · ${actual}`
  if (h.append_succeeded !== true || !h.executed_mode) return '交接结果待确认'
  if (!/^(token|kv(?::v\d+)?)$/i.test(h.executed_mode))
    return `交接模式未确认（${actual}）`
  return `${h.fallback_reason ? '回退接收' : '接收'} ${actual}`
}
export function receptionForTurn(
  session: DiscussionSession,
  turn: Turn,
): { label: string; handoff?: Handoff; failed?: boolean } {
  if (!turn.received_message_id) return { label: turn.sequence === 1 && turn.agent === 'A' && turn.round === 1 ? '初始发言' : '接收来源待确认' }
  const previous = session.turns.find(t => t.message_id === turn.received_message_id)
  const receipts = session.handoffs.filter(h => h.message_id === turn.received_message_id && h.target === turn.agent && (!previous || (h.source === previous.agent && h.discussion_round === previous.round)))
  if (receipts.length !== 1) return { label: '接收来源待确认' }
  const h = receipts[0]
  return {
    label: receiptLabel(h),
    handoff: h,
    failed: !!h.error || h.append_succeeded === false,
  }
}
export function deliveryForTurn(session: DiscussionSession, turn: Turn): { label: string; handoff?: Handoff; failed?: boolean } {
  const receipts = session.handoffs.filter(h => h.message_id === turn.message_id && h.source === turn.agent && (!turn.target || h.target === turn.target))
  if (receipts.length === 1) {
    const h = receipts[0]
    return { label: receiverCommitted(h) && h.executed_mode && !h.error ? `${h.target} 已${receiptLabel(h)}` : receiptLabel(h), handoff: h, failed: !!h.error || h.append_succeeded === false }
  }
  const active = session.active_handoff
  const decision = session.auto_decisions?.find(d => d.message_id === turn.message_id)
  if (decision?.decision_status === 'failed') return { label: 'AUTO 决策失败 · 未交付', failed: true }
  if (active?.message_id === turn.message_id && session.architecture === 'distributed') return { label: '发送中 · 等待 Receiver 提交确认' }
  if (active?.message_id === turn.message_id) return { label: active.error ? '交接失败 · 未交付' : `请求 ${modeLabel(active.requested_mode ?? (active.decision ? 'auto' : undefined))} · 选择 ${modeLabel(active.decision?.selected_candidate ?? active.auto_decision?.selected_candidate ?? active.selected_mode, active.selected_kv_strategy)} · 尚未交付`, failed: !!active.error }
  return { label: receipts.length > 1 ? '交付记录存在歧义' : '已生成 · 尚无关联交付记录' }
}
export function discussionStage(session: DiscussionSession | null) {
  const last = session?.handoffs.at(-1)
  const pending =
    session?.status === 'running' && ['HANDOFF_STARTED', 'AUTO_DECISION'].includes(session.phase)
  const source = pending ? session.agent : last?.source
  const target = pending
    ? source === 'A'
      ? 'B'
      : source === 'B'
        ? 'A'
        : null
    : last?.target
  const states = ['A', 'B'].map((name) => ({
    name: `Agent ${name}`,
    state: !session
      ? '等待开始'
      : session.status === 'delivery_unknown'
        ? '提交结果未知'
      : session.status === 'failed'
        ? '实验已停止'
        : session.status === 'completed'
          ? '已完成发言'
          : pending
            ? name === source
              ? '正在交接消息'
              : '等待接收'
            : session.agent === name
              ? session.phase === 'DISCUSSION_FINALIZING'
                ? '正在生成总结'
                : session.phase === 'AGENT_TURN_STARTED'
                  ? '正在生成发言'
                  : '发言已生成'
              : '等待对方',
  }))
  return {
    agents: states,
    source,
    target,
    pending,
    status: session ? (session.status === 'delivery_unknown' ? '提交结果未知 · 需对账' : phaseNames[session.phase] ?? session.phase) : '准备开始',
    mode: pending
      ? session?.active_handoff
        ? `本次交接 · 请求 ${modeLabel(session.active_handoff.requested_mode ?? 'auto')} · 选择 ${modeLabel(session.active_handoff.decision?.selected_candidate ?? session.active_handoff.auto_decision?.selected_candidate ?? session.active_handoff.selected_mode, session.active_handoff.selected_kv_strategy)} · 尚未交付`
        : '本次交接 · 模式待确认'
      : last
        ? `最近交接 · ${receiptLabel(last)}`
        : '尚无交接',
    repair:
      !pending && last?.executed_mode?.toLowerCase().startsWith('kv')
        ? (last.kv_repair_method ?? '未报告')
        : null,
    handoff: pending ? undefined : last,
  }
}

export function latestReceptionForAgent(
  session: DiscussionSession,
  agent: string,
) {
  const receipt = session.handoffs
    .filter((h) => h.target === agent)
    .sort((a, b) => b.sequence - a.sequence)[0]
  return receipt
    ? { label: `上下文最近${receiptLabel(receipt)}`, handoff: receipt }
    : null
}
