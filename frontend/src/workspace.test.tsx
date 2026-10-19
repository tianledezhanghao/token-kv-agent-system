import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { baseSnapshot } from './test/fixtures'
class Socket {
  onopen = null
  onclose = null
  onmessage = null
  onerror = null
  close() {}
}
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})
describe('Workspace state isolation', () => {
  it('keeps discussion draft and received data when switching to canonical and back', async () => {
    const fetch = vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input)
      const data =
        path === '/api/discussions/latest'
          ? { available: true, session: null, model: null }
          : path === '/api/discussions/prompts'
            ? {
                agent_a: { text: 'default A', source: 'default' },
                agent_b: { text: 'default B', source: 'default' },
                version: 'test',
                stages: {},
              }
            : path === '/api/health'
              ? { status: 'ok' }
              : path === '/api/state'
                ? baseSnapshot
                : { events: [], count: 0 }
      return { ok: true, json: async () => data } as Response
    })
    vi.stubGlobal('fetch', fetch)
    vi.stubGlobal('WebSocket', Socket)
    render(<App />)
    await waitFor(() =>
      expect(screen.getByRole('button', { name: '开始讨论' })).toBeEnabled(),
    )
    fireEvent.change(screen.getByLabelText('实验问题'), {
      target: { value: 'Keep this draft' },
    })
    fireEvent.click(screen.getByRole('button', { name: '通信基准' }))
    await within(
      screen.getByRole('main', { name: '通信基准工作区' }),
    ).findByRole('heading', { name: '通信基准' })
    fireEvent.click(screen.getByRole('button', { name: '双 Agent 讨论' }))
    expect(screen.getByLabelText('实验问题')).toHaveValue('Keep this draft')
    expect(
      fetch.mock.calls.filter((c) => c[0] === '/api/discussions'),
    ).toHaveLength(0)
  })
})
