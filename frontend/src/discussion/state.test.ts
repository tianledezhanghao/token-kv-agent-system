import { describe, expect, it } from 'vitest'
import { projectDiscussion, restoreDiscussion } from './state'
import type { DashboardEvent } from '../types/dashboard'
import type { DiscussionSession } from './types'
const state = {
  session_id: 'session-1',
  created_at: '2026-10-10T01:00:00Z',
  revision: 4,
  turns: [],
  handoffs: [],
  phase: 'HANDOFF_STARTED',
  status: 'running',
} as unknown as DiscussionSession
const event = (metadata: Record<string, unknown>, type = 'HANDOFF_COMPLETED') =>
  ({ type, timestamp: '2026-10-10T01:01:00Z', metadata }) as DashboardEvent
describe('Discussion state ordering', () => {
  it.each(['AUTO_DECISION', 'AUTO_DECISION_FAILED'])('projects %s without losing revision order', (type) => {
    const next = projectDiscussion(state, event({ session_id: 'session-1', revision: 5, status: 'running', sequence: 1 }, type))
    expect(next?.revision).toBe(5)
    expect(next?.phase).toBe(type)
    expect(next?.handoffs).toEqual([])
  })
  it('waits for REST on a revision gap', () => {
    expect(
      projectDiscussion(state, event({ session_id: 'session-1', revision: 6 })),
    ).toBe(state)
  })
  it('ignores malformed revision values', () => {
    expect(
      projectDiscussion(
        state,
        event({ session_id: 'session-1', revision: '5' }),
      ),
    ).toBe(state)
  })

  it('ignores stale or duplicate revisions', () => {
    expect(
      projectDiscussion(state, event({ session_id: 'session-1', revision: 4 })),
    ).toBe(state)
  })
  it('ignores events for another session', () => {
    expect(
      projectDiscussion(state, event({ session_id: 'session-2', revision: 5 })),
    ).toBe(state)
  })
  it('ignores Canonical events', () => {
    expect(projectDiscussion(state, event({}, 'TRANSMISSION_FINISHED'))).toBe(
      state,
    )
  })
  it('older REST cannot roll back WS state', () => {
    expect(restoreDiscussion(state, { ...state, revision: 3 })).toBe(state)
  })
  it('authoritative same-revision REST repairs missing deltas', () => {
    const repaired = {
      ...state,
      handoffs: [{ sequence: 1 }],
    } as DiscussionSession
    expect(restoreDiscussion(state, repaired)).toBe(repaired)
  })
  it('rejects old-session REST responses after a new start', () => {
    expect(
      restoreDiscussion(state, {
        ...state,
        session_id: 'older',
        created_at: '2026-10-10T00:00:00Z',
      }),
    ).toBe(state)
  })
  it('accepts a newer session', () => {
    const newer = {
      ...state,
      session_id: 'newer',
      created_at: '2026-10-10T02:00:00Z',
    }
    expect(restoreDiscussion(state, newer)).toBe(newer)
  })
})

it('cannot project a Local event into the same-named Distributed session', () => {
  const distributed = { ...state, architecture: 'distributed' } as DiscussionSession
  expect(projectDiscussion(distributed,event({session_id:'session-1',architecture:'local',revision:5}))).toBe(distributed)
})
it('deduplicates a distributed handoff by index without adding Agent messages', () => {
  const distributed = { ...state, architecture: 'distributed', handoffs: [{sequence:1}] } as DiscussionSession
  const next=projectDiscussion(distributed,event({session_id:'session-1',architecture:'distributed',revision:5,handoff:{sequence:1},status:'delivery_unknown'}))
  expect(next?.handoffs).toHaveLength(1)
  expect(next?.turns).toHaveLength(0)
  expect(next?.status).toBe('delivery_unknown')
})
it('rejects same-ID REST records from a different architecture', () => {
  expect(restoreDiscussion(state,{...state,architecture:'distributed',revision:20})).toBe(state)
})
