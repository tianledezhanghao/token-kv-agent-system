import { useEffect, useRef, useState } from 'react'
import { eventWebSocketUrl } from '../api/dashboard'
import type { ConnectionStatus, DashboardEvent } from '../types/dashboard'
interface EventStreamOptions {
  path?: string
  enabled: boolean
  onEvent: (event: DashboardEvent) => void
  onReconnect: () => void
}

export function useEventStream({
  enabled,
  path,
  onEvent,
  onReconnect,
}: EventStreamOptions): ConnectionStatus {
  const [status, setStatus] = useState<ConnectionStatus>('disconnected')
  const onEventRef = useRef(onEvent),
    onReconnectRef = useRef(onReconnect)
  useEffect(() => {
    onEventRef.current = onEvent
    onReconnectRef.current = onReconnect
  }, [onEvent, onReconnect])
  useEffect(() => {
    if (!enabled) return
    let socket: WebSocket | null = null,
      retryTimer: number | undefined,
      stopped = false,
      retryCount = 0,
      generation = 0
    const clearRetry = () => {
      if (retryTimer !== undefined) window.clearTimeout(retryTimer)
      retryTimer = undefined
    }
    const dispose = () => {
      const old = socket
      socket = null
      if (old) {
        old.onopen = null
        old.onmessage = null
        old.onerror = null
        old.onclose = null
        old.close()
      }
    }
    const connect = () => {
      if (stopped) return
      clearRetry()
      generation += 1
      const currentGeneration = generation
      dispose()
      if (!navigator.onLine) {
        setStatus('disconnected')
        return
      }
      setStatus('reconnecting')
      const current = new WebSocket(path ? eventWebSocketUrl().replace('/ws/events', path) : eventWebSocketUrl())
      socket = current
      const active = () => !stopped && generation === currentGeneration
      current.onopen = () => {
        if (!active()) return
        retryCount = 0
        setStatus('connected')
        onReconnectRef.current()
      }
      current.onmessage = (message) => {
        if (!active()) return
        try {
          const event = JSON.parse(String(message.data)) as DashboardEvent
          if (
            event &&
            typeof event.type === 'string' &&
            event.metadata &&
            typeof event.metadata === 'object' &&
            !Array.isArray(event.metadata)
          )
            onEventRef.current(event)
        } catch {
          /* Ignore malformed frames. REST recovery remains authoritative. */
        }
      }
      current.onerror = () => {
        if (active()) current.close()
      }
      current.onclose = () => {
        if (!active()) return
        if (!navigator.onLine) {
          setStatus('disconnected')
          return
        }
        retryCount += 1
        setStatus('reconnecting')
        clearRetry()
        retryTimer = window.setTimeout(
          connect,
          Math.min(2000 * retryCount, 10000),
        )
      }
    }
    const offline = () => {
      clearRetry()
      generation += 1
      dispose()
      setStatus('disconnected')
    }
    const online = () => connect()
    window.addEventListener('offline', offline)
    window.addEventListener('online', online)
    connect()
    return () => {
      stopped = true
      generation += 1
      clearRetry()
      window.removeEventListener('offline', offline)
      window.removeEventListener('online', online)
      dispose()
    }
  }, [enabled, path])
  return status
}
