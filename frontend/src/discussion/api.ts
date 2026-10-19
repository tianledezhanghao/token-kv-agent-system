import type { Architecture, NetworkPatch } from './types'
import type { DiscussionConfig, DiscussionSession, LatestDiscussion, PromptConfig, NetworkUpdate } from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json' } })
  const body = await response.json()
  if (!response.ok) {
    const detail = body?.detail
    throw new Error(Array.isArray(detail) ? detail.map((item: { msg: string }) => item.msg).join('; ') : String(detail ?? `HTTP ${response.status}`))
  }
  return body as T
}
const prefix = (a: Architecture = 'local') => a === 'distributed' ? '/api/distributed' : '/api'
export const getDiscussion = (sessionId: string, a?: Architecture) => request<DiscussionSession>(`${prefix(a)}/discussions/${encodeURIComponent(sessionId)}`)
export const getLatestDiscussion = (a?: Architecture) => request<LatestDiscussion>(`${prefix(a)}/discussions/latest`)
export const previewDiscussionPrompts = (agentA: string, agentB: string, a?: Architecture) =>
  request<PromptConfig>(`${prefix(a)}/discussions/prompts`, { method: 'POST', body: JSON.stringify({ agent_a_system_prompt: agentA, agent_b_system_prompt: agentB }) })
export const startDiscussion = (config: DiscussionConfig, sessionId: string) =>
  request<DiscussionSession>(`${prefix(config.architecture)}/discussions`, { method: 'POST', body: JSON.stringify({ ...config, session_id: sessionId }) })

export const patchDiscussionNetwork = (id: string, config: NetworkPatch, a?: Architecture) =>
  request<NetworkUpdate & { updated: boolean; revision: number }>(`${prefix(a)}/discussions/${encodeURIComponent(id)}/network`, { method: 'PATCH', body: JSON.stringify(config) })

export const switchModel = (model: string) => request<LatestDiscussion>('/api/models/switch', { method: 'POST', body: JSON.stringify({ model }) })

export const releaseDiscussion = (id: string) => request(`/api/distributed/discussions/${encodeURIComponent(id)}/release`, { method: 'POST' })
