import type { DashboardEvent } from '../types/dashboard'
import type { ActiveHandoff, AutoDecision, DecisionRecord, DiscussionSession, Handoff, Turn, NetworkState, NetworkUpdate } from './types'

export function projectDiscussion(
  session: DiscussionSession | null,
  event: DashboardEvent,
): DiscussionSession | null {
  const m = event.metadata
  if (
    !event.type.startsWith('DISCUSSION_') &&
    ![
      'AGENT_TURN_STARTED',
      'AGENT_TURN_COMPLETED',
      'AUTO_DECISION',
      'AUTO_DECISION_FAILED',
      'HANDOFF_STARTED',
      'HANDOFF_COMPLETED',
      'HANDOFF_FAILED',
      'HANDOFF_UNCONFIRMED',
    ].includes(event.type)
  )
    return session
  if (
    !session ||
    m.session_id !== session.session_id ||
    (m.architecture ?? 'local') !== (session.architecture ?? 'local') ||
    !Number.isInteger(m.revision) ||
    Number(m.revision) !== session.revision + 1
  )
    return session
  const next = {
    ...session,
    revision: Number(m.revision),
    status: m.status as DiscussionSession['status'],
    phase: event.type === 'DISCUSSION_NETWORK_UPDATED' ? session.phase : event.type,
    round: Number(m.round),
    agent: m.agent as string | null,
    sequence: Number(m.sequence),
    updated_at: event.timestamp,
  }
  if (event.type === 'DISCUSSION_NETWORK_UPDATED') {
    next.network = m.network as unknown as NetworkState
    next.network_updates = [...(session.network_updates ?? []), m.network_update as unknown as NetworkUpdate]
  }
  if (m.network_state) next.network = m.network_state as unknown as NetworkState
  if (m.turn) next.turns = [...session.turns.filter(t => t.message_id !== (m.turn as unknown as Turn).message_id), m.turn as unknown as Turn]
  if (m.decision || event.type === 'AUTO_DECISION_FAILED') {
    const record: DecisionRecord = m.decision
      ? { ...(m.decision as unknown as AutoDecision), handoff_index: Number(m.sequence), message_id: m.message_id as string, source: m.agent as string, target: m.target as string }
      : m as unknown as DecisionRecord
    next.auto_decisions = [...(session.auto_decisions ?? []).filter(d => (d.handoff_index ?? d.sequence) !== Number(m.sequence)), record]
  }
  if (['AUTO_DECISION', 'AUTO_DECISION_FAILED', 'HANDOFF_STARTED'].includes(event.type)) next.active_handoff = m as unknown as ActiveHandoff
  if (['HANDOFF_COMPLETED', 'HANDOFF_FAILED', 'HANDOFF_UNCONFIRMED', 'DISCUSSION_COMPLETED', 'DISCUSSION_FAILED'].includes(event.type)) next.active_handoff = null
  if (m.handoff)
    next.handoffs = [...session.handoffs.filter(h => h.sequence !== (m.handoff as unknown as Handoff).sequence), m.handoff as unknown as Handoff]
  for (const key of [
    'score',
    'final_answer',
    'summary',
    'error',
    'failure_stage',
    'termination_reason',
    'cache_lengths',
    'cache_released',
  ] as const) {
    if (key in m) Object.assign(next, { [key]: m[key] })
  }
  return next
}

export function restoreDiscussion(
  current: DiscussionSession | null,
  incoming: DiscussionSession | null,
): DiscussionSession | null {
  if (!incoming) return current // Process reset is handled explicitly by the hook.
  if (!current) return incoming
  if (current.session_id === incoming.session_id && (current.architecture ?? 'local') !== (incoming.architecture ?? 'local')) return current
  if (current.session_id === incoming.session_id)
    return incoming.revision >= current.revision ? incoming : current
  return incoming.created_at >= current.created_at ? incoming : current
}
