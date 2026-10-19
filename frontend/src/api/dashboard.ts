import type {
  CommunicationControlResponse,
  DashboardSnapshot,
  EventsResponse,
  NetworkState,
} from '../types/dashboard'

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { Accept: 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null
    throw new Error(body?.detail ?? `${path} returned ${response.status}`)
  }
  return response.json() as Promise<T>
}

const getJson = <T,>(path: string) => requestJson<T>(path)

export async function checkHealth(): Promise<void> {
  const health = await getJson<{ status: string }>('/api/health')
  if (health.status !== 'ok') {
    throw new Error('Dashboard Backend is not healthy')
  }
}

export const getDashboardState = () => getJson<DashboardSnapshot>('/api/state')

export const getRecentEvents = (limit = 100) =>
  getJson<EventsResponse>(`/api/events?limit=${limit}`)

export const updateNetwork = (bandwidthMbps: number) =>
  requestJson<{ network: NetworkState }>('/api/control/network', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bandwidth_mbps: bandwidthMbps }),
  })

export const triggerCommunication = () =>
  requestJson<CommunicationControlResponse>('/api/control/communication', {
    method: 'POST',
  })

export function eventWebSocketUrl(): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/ws/events`
}
