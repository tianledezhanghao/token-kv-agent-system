import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useEventStream } from './useEventStream'
class Socket {
  static instances: Socket[] = []
  onopen: (() => void) | null = null
  onmessage: ((e: MessageEvent) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null
  close = vi.fn()
  constructor() {
    Socket.instances.push(this)
  }
}
const receive = vi.fn(),
  recover = vi.fn()
function Stream() {
  const status = useEventStream({
    enabled: true,
    onEvent: receive,
    onReconnect: recover,
  })
  return <p>{status}</p>
}
beforeEach(() => {
  Socket.instances = []
  vi.useFakeTimers()
  vi.stubGlobal('WebSocket', Socket)
  receive.mockClear()
  recover.mockClear()
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})
describe('WebSocket lifecycle', () => {
  it('recovers the REST/WS initial connection gap and a later reconnect', () => {
    render(<Stream />)
    act(() => Socket.instances[0].onopen?.())
    expect(recover).toHaveBeenCalledTimes(1)
    act(() => Socket.instances[0].onclose?.())
    act(() => vi.advanceTimersByTime(2000))
    expect(Socket.instances).toHaveLength(2)
    act(() => Socket.instances[1].onopen?.())
    expect(recover).toHaveBeenCalledTimes(2)
    expect(screen.getByText('connected')).toBeInTheDocument()
  })
  it('clears pending retries on offline and opens only one replacement on online', () => {
    render(<Stream />)
    act(() => Socket.instances[0].onclose?.())
    act(() => window.dispatchEvent(new Event('offline')))
    expect(screen.getByText('disconnected')).toBeInTheDocument()
    act(() => vi.advanceTimersByTime(20000))
    expect(Socket.instances).toHaveLength(1)
    act(() => window.dispatchEvent(new Event('online')))
    expect(Socket.instances).toHaveLength(2)
    act(() => vi.advanceTimersByTime(20000))
    expect(Socket.instances).toHaveLength(2)
  })
  it('ignores old connection callbacks and malformed frames', () => {
    render(<Stream />)
    const old = Socket.instances[0].onmessage
    act(() => window.dispatchEvent(new Event('online')))
    act(() =>
      old?.({
        data: JSON.stringify({ type: 'LATE', metadata: {} }),
      } as MessageEvent),
    )
    const current = Socket.instances[1]
    act(() => current.onmessage?.({ data: '{broken' } as MessageEvent))
    act(() =>
      current.onmessage?.({
        data: JSON.stringify({ type: 'BAD' }),
      } as MessageEvent),
    )
    expect(receive).not.toHaveBeenCalled()
    act(() =>
      current.onmessage?.({
        data: JSON.stringify({ type: 'OK', metadata: {} }),
      } as MessageEvent),
    )
    expect(receive).toHaveBeenCalledTimes(1)
  })
  it('detaches socket handlers and timers on unmount', () => {
    const view = render(<Stream />)
    const old = Socket.instances[0]
    act(() => old.onclose?.())
    view.unmount()
    act(() => vi.advanceTimersByTime(20000))
    expect(Socket.instances).toHaveLength(1)
    expect(old.onopen).toBeNull()
    expect(old.onmessage).toBeNull()
    expect(old.close).toHaveBeenCalled()
  })
})
