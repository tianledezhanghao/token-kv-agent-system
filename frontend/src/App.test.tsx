import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import CanonicalDashboard from './canonical/CanonicalDashboard'
import {
  baseSnapshot,
  realKVSnapshot,
  realTokenSnapshot,
  tokenSelectionEvent,
} from './test/fixtures'
import type { DashboardEvent, DashboardSnapshot } from './types/dashboard'
class MockWebSocket {
  static instances: MockWebSocket[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null
  constructor(public url: string) {
    MockWebSocket.instances.push(this)
  }
  close() {}
  emit(event: DashboardEvent) {
    this.onmessage?.({ data: JSON.stringify(event) } as MessageEvent)
  }
}
const response = (value: unknown, ok = true) =>
  Promise.resolve({
    ok,
    status: ok ? 200 : 503,
    json: async () => value,
  } as Response)
function serve(
  snapshot: DashboardSnapshot = baseSnapshot,
  events: DashboardEvent[] = [tokenSelectionEvent],
  control?: (path: string, init?: RequestInit) => Promise<Response> | undefined,
) {
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input)
    if (path === '/api/health') return response({ status: 'ok' })
    if (path === '/api/state') return response(snapshot)
    if (path.startsWith('/api/events'))
      return response({ events, count: events.length })
    return control?.(path, init) ?? response({ detail: 'not available' }, false)
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}
const openConfig = () =>
  fireEvent.click(screen.getByRole('button', { name: '配置' }))
const closeModal = () => {
  const d = screen.getByRole('dialog')
  fireEvent.click(within(d).getByRole('button', { name: /^关闭/ }))
}
const openDetails = (tab = '成本') => {
  fireEvent.click(screen.getByRole('button', { name: '实验详情' }))
  if (tab !== '成本')
    fireEvent.click(
      within(screen.getByRole('dialog')).getByRole('button', { name: tab }),
    )
}
async function ready() {
  render(<CanonicalDashboard />)
  await screen.findByText(/最近交接/)
}
function adaptiveSnapshot(selected: 'TOKEN' | 'KV'): DashboardSnapshot {
  return {
    ...baseSnapshot,
    runtime: {
      mode: 'real_adaptive',
      running: false,
      next_source: 'Agent B',
      next_target: 'Agent A',
      active_communication_mode: selected,
      selector_status: 'active',
      selector_kind: 'real_cost_based',
      cost_boundary: 'MESSAGE_READY',
      selector_profile: {
        profile_name: 'macbook-development',
        device_type: 'mps',
        model_id: 'test/model',
        cost_boundary: 'MESSAGE_READY',
        calibration_token_counts: [8, 32, 128],
        development_calibration: true,
      },
    },
    last_selection: {
      source: 'Agent A',
      target: 'Agent B',
      mode: selected,
      selected_mode: selected,
      token_count: 8,
      bandwidth_mbps: 20,
      base_latency_ms: 5,
      cost_boundary: 'MESSAGE_READY',
      selector_kind: 'real_cost_based',
      estimated_saving_ms: 21.22,
      estimated_saving_ratio: 0.5,
      estimated_token_latency_ms: 21.09,
      estimated_kv_latency_ms: 42.31,
      token_estimate: {
        wire_bytes: 42,
        serialize_ms: 0.01,
        link_ms: 5.02,
        deserialize_ms: 0.01,
        receiver_prefill_ms: 16.05,
        total_ms: 21.09,
      },
      kv_estimate: {
        raw_kv_bytes: 98304,
        wire_metadata_bytes: 2283,
        wire_bytes: 100587,
        sender_prepare_ms: 16.34,
        serialize_ms: 9.85,
        link_ms: 45.23,
        deserialize_ms: 0.1,
        restore_ms: 10.19,
        message_ready_total_ms: 42.31,
      },
    },
    adaptive_execution: {
      source: 'Agent A',
      target: 'Agent B',
      executed_mode: selected,
      predicted_ms: selected === 'TOKEN' ? 21.09 : 42.31,
      actual_ms: 24.2,
      absolute_error_ms: 3.11,
      relative_error: 0.147,
      sender_text: 'sender',
      receiver_text: 'receiver',
      correctness: true,
      selected_measurement: {
        wire_bytes: selected === 'TOKEN' ? 42 : 100587,
        raw_kv_bytes: 98304,
        wire_metadata_bytes: 2283,
        sender_prepare_ms: 16,
        device_to_host_ms: 3,
        encode_ms: 4,
        serialize_ms: 0.01,
        link_ms: 5.02,
        deserialize_ms: 0.01,
        cpu_reconstruction_ms: 2,
        host_to_device_ms: 3,
        cache_construction_ms: 1,
        restore_ms: 10,
        receiver_prefill_ms: 16,
        receiver_prefill_skipped: true,
      },
    },
    recent_events: [
      {
        ...tokenSelectionEvent,
        type: 'REAL_SELECTION_MADE',
        metadata: {
          token_estimate: { total_ms: 21.09 },
          kv_estimate: { message_ready_total_ms: 42.31 },
          selected_mode: selected,
        },
      },
    ],
  }
}
beforeEach(() => {
  MockWebSocket.instances = []
  vi.stubGlobal('WebSocket', MockWebSocket)
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('Canonical workspace functional migration', () => {
  it.each(['TOKEN', 'KV'] as const)(
    'keeps adaptive %s prediction distinct from the single actual path',
    async (selected) => {
      serve(adaptiveSnapshot(selected), [])
      await ready()
      openDetails()
      expect(screen.getByText('Real Adaptive Decision')).toBeInTheDocument()
      expect(screen.getByText('TOKEN Estimated')).toBeInTheDocument()
      expect(screen.getByText('KV Estimated')).toBeInTheDocument()
      expect(screen.getByText(`Selected ${selected}`)).toBeInTheDocument()
      expect(
        screen.getByText(
          `${selected === 'TOKEN' ? 'KV' : 'TOKEN'} Actual: Not executed`,
        ),
      ).toBeInTheDocument()
      expect(screen.getByText('Absolute Error')).toBeInTheDocument()
      if (selected === 'KV')
        expect(screen.getByText('Skipped')).toBeInTheDocument()
      closeModal()
      openConfig()
      expect(screen.getByText('macbook-development')).toBeInTheDocument()
      expect(screen.getByText('Active · Real Cost Based')).toBeInTheDocument()
    },
  )
  it('retains network, selector estimates, recent bytes and cost fields on demand', async () => {
    serve()
    await ready()
    openConfig()
    expect(screen.getByLabelText('带宽 Mbps')).toHaveValue(20)
    closeModal()
    openDetails()
    expect(screen.getAllByText('25.4 ms').length).toBeGreaterThan(0)
    expect(screen.getByText('405 ms')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '最近通信' }))
    expect(screen.getByText('1 KB')).toBeInTheDocument()
    expect(
      within(screen.getByRole('dialog')).getByText('Agent A'),
    ).toBeInTheDocument()
  })
  it('retains historical log summaries and full metadata behind details', async () => {
    serve()
    await ready()
    openDetails('日志')
    expect(
      screen.getByText('模式选择完成', { selector: 'strong' }),
    ).toBeInTheDocument()
    expect(screen.getByText(/TOKEN 25.4 ms/)).toBeInTheDocument()
    expect(screen.getByText(/"estimated_token_latency_ms"/)).toBeInTheDocument()
  })
  it('shows a reverse actual KV route from REST', async () => {
    serve({
      ...baseSnapshot,
      last_communication: {
        ...baseSnapshot.last_communication!,
        source: 'Agent B',
        target: 'Agent A',
        mode: 'KV',
        payload_bytes: 1000000,
        transmission_latency_ms: 13,
        estimated_e2e_latency_ms: 13,
      },
    })
    await ready()
    expect(screen.getByLabelText('Agent B → Agent A')).toBeInTheDocument()
    expect(document.querySelector('.wire-arrow.reverse')).toBeInTheDocument()
    openDetails('最近通信')
    expect(screen.getByText('1 MB')).toBeInTheDocument()
  })
  it('projects live network updates', async () => {
    serve()
    await ready()
    openConfig()
    act(() =>
      MockWebSocket.instances[0].emit({
        ...tokenSelectionEvent,
        type: 'NETWORK_STATE_CHANGED',
        metadata: { bandwidth_mbps: 1000, base_latency_ms: 5 },
      }),
    )
    expect(screen.getByLabelText('带宽 Mbps')).toHaveValue(1000)
  })
  it('shows connected state after socket opens and refreshes the initial race window', async () => {
    const fetch = serve()
    await ready()
    act(() => MockWebSocket.instances[0].onopen?.())
    expect(screen.getByText('已连接')).toBeInTheDocument()
    await waitFor(() =>
      expect(
        fetch.mock.calls.filter((c) => c[0] === '/api/state'),
      ).toHaveLength(2),
    )
  })
  it('keeps absent selection and communication as empty rather than zeros', async () => {
    serve(
      { ...baseSnapshot, last_selection: null, last_communication: null },
      [],
    )
    await ready()
    expect(screen.getByText('尚无 Agent 生成内容。')).toBeInTheDocument()
    openDetails()
    expect(screen.getAllByText('—').length).toBeGreaterThan(0)
    fireEvent.click(screen.getByRole('button', { name: '最近通信' }))
    expect(screen.getByText('尚无通信记录。')).toBeInTheDocument()
  })
  it('keeps a failed backend actionable without a crash', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(new Error('offline'))),
    )
    render(<CanonicalDashboard />)
    expect(await screen.findByRole('alert')).toHaveTextContent('后端当前不可用')
    expect(screen.getByRole('button', { name: '重新同步' })).toBeInTheDocument()
  })
  it('sends only a positive bandwidth input to PATCH and marks unapplied drafts', async () => {
    const fetch = serve(baseSnapshot, [], (path) =>
      path === '/api/control/network'
        ? response({ network: { bandwidth_mbps: 1000, base_latency_ms: 5 } })
        : undefined,
    )
    await ready()
    openConfig()
    fireEvent.change(screen.getByLabelText('带宽 Mbps'), {
      target: { value: '1000' },
    })
    expect(screen.getByText(/有未应用的带宽草稿/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '应用网络设置' }))
    await waitFor(() =>
      expect(fetch).toHaveBeenCalledWith(
        '/api/control/network',
        expect.objectContaining({
          method: 'PATCH',
          body: JSON.stringify({ bandwidth_mbps: 1000 }),
        }),
      ),
    )
    expect(
      fetch.mock.calls.filter((c) => c[0] === '/api/control/network'),
    ).toHaveLength(1)
  })
  it('rejects invalid bandwidth locally', async () => {
    const fetch = serve()
    await ready()
    openConfig()
    fireEvent.change(screen.getByLabelText('带宽 Mbps'), {
      target: { value: '0' },
    })
    expect(screen.getByRole('button', { name: '应用网络设置' })).toBeDisabled()
    expect(
      fetch.mock.calls.filter((c) => c[0] === '/api/control/network'),
    ).toHaveLength(0)
  })
  it('disables communication and rejects repeat POSTs until completion', async () => {
    let finish!: (r: Response) => void
    const pending = new Promise<Response>((resolve) => {
      finish = resolve
    })
    const fetch = serve(baseSnapshot, [], (path) =>
      path === '/api/control/communication' ? pending : undefined,
    )
    await ready()
    const button = screen.getByRole('button', { name: '运行一次通信' })
    fireEvent.click(button)
    fireEvent.click(button)
    expect(screen.getByRole('button', { name: '通信运行中…' })).toBeDisabled()
    expect(
      fetch.mock.calls.filter((c) => c[0] === '/api/control/communication'),
    ).toHaveLength(1)
    await act(async () =>
      finish(
        await response({
          source: 'Agent A',
          target: 'Agent B',
          selected_mode: 'TOKEN',
          runtime: baseSnapshot.runtime,
        }),
      ),
    )
    expect(screen.getByRole('button', { name: '运行一次通信' })).toBeEnabled()
  })
  it('projects high-bandwidth selector KV as a decision, leaving last actual receipt distinct', async () => {
    serve()
    await ready()
    act(() =>
      MockWebSocket.instances[0].emit({
        ...tokenSelectionEvent,
        source: 'Agent B',
        target: 'Agent A',
        mode: 'KV',
        metadata: {
          estimated_token_latency_ms: 25.008,
          estimated_kv_latency_ms: 13,
        },
      }),
    )
    openDetails()
    expect(screen.getByText('13 ms')).toBeInTheDocument()
    expect(screen.getAllByText('KV CACHE').length).toBeGreaterThan(0)
  })
  it('shows a 409 communication failure inline', async () => {
    serve(baseSnapshot, [], (path) =>
      path === '/api/control/communication'
        ? response({ detail: 'communication is already running' }, false)
        : undefined,
    )
    await ready()
    fireEvent.click(screen.getByRole('button', { name: '运行一次通信' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('通信正在运行')
  })
  it('retains bandwidth draft when PATCH fails', async () => {
    serve(baseSnapshot, [], (path) =>
      path === '/api/control/network'
        ? response({ detail: 'network update failed' }, false)
        : undefined,
    )
    await ready()
    openConfig()
    fireEvent.change(screen.getByLabelText('带宽 Mbps'), {
      target: { value: '1000' },
    })
    fireEvent.click(screen.getByRole('button', { name: '应用网络设置' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('网络更新失败')
    expect(screen.getByLabelText('带宽 Mbps')).toHaveValue(1000)
  })
  it.each(['auto', 'observer'] as const)(
    'keeps %s read-only and does not offer an invalid mode switch',
    async (mode) => {
      serve({ ...baseSnapshot, runtime: { ...baseSnapshot.runtime, mode } })
      await ready()
      expect(
        screen.queryByRole('button', { name: '运行一次通信' }),
      ).not.toBeInTheDocument()
      openConfig()
      expect(
        screen.getByText(mode === 'auto' ? '自动演示模式' : '只读观察模式'),
      ).toBeInTheDocument()
      expect(screen.queryByLabelText('带宽 Mbps')).not.toBeInTheDocument()
    },
  )
  it('keeps real TOKEN measurements and content without KV predictions', async () => {
    serve(realTokenSnapshot, [])
    await ready()
    expect(
      screen.getByText('A cache stores reusable attention state.'),
    ).toBeInTheDocument()
    expect(
      screen.getByText('It avoids recomputing earlier tokens.'),
    ).toBeInTheDocument()
    openDetails()
    expect(screen.getByText('Token 成本明细')).toBeInTheDocument()
    expect(screen.queryByText('KV Estimated')).not.toBeInTheDocument()
    for (const t of [
      '0.001 ms',
      '0.006 ms',
      '15.124 ms',
      '20.148 ms',
      'Verified',
    ])
      expect(screen.getByText(t)).toBeInTheDocument()
    expect(screen.getByText('42 B')).toBeInTheDocument()
    closeModal()
    openConfig()
    expect(screen.getByText('Qwen/Qwen2.5-0.5B-Instruct')).toBeInTheDocument()
    expect(screen.getByText('mps')).toBeInTheDocument()
  })
  it('runs fixed TOKEN without claiming adaptive selection', async () => {
    serve(realTokenSnapshot, [], (path) =>
      path === '/api/control/communication'
        ? response({
            source: 'Agent B',
            target: 'Agent A',
            communication_mode: 'TOKEN',
            selected_mode: null,
            runtime: realTokenSnapshot.runtime,
          })
        : undefined,
    )
    await ready()
    fireEvent.click(screen.getByRole('button', { name: '运行一次通信' }))
    expect(
      await screen.findByText('Agent B → Agent A 真实 Token handoff 完成。'),
    ).toBeInTheDocument()
    expect(screen.queryByText(/Selector 自动选择/)).not.toBeInTheDocument()
  })
  it('keeps all KV preparation/restore/correctness boundaries and raw diagnostics', async () => {
    serve(realKVSnapshot, [])
    await ready()
    openDetails()
    for (const t of [
      'KV 通信成本',
      'Raw KV',
      'Metadata Overhead',
      'Sender KV Preparation',
      'Device → Host',
      'Encode',
      'Total Serialize',
      'Simulated Link',
      'Deserialize',
      'CPU Reconstruction',
      'Host → Device',
      'Cache Construction',
      'Total Restore',
      'Representation-ready',
      'Message-ready E2E',
      'Skipped — restored from KV cache',
      'Verified',
    ])
      expect(screen.getByText(t)).toBeInTheDocument()
    expect(screen.getAllByText('98.3 KB').length).toBeGreaterThan(0)
    expect(screen.getByText('2.3 KB')).toBeInTheDocument()
    expect(screen.getByText('60.335 ms')).toBeInTheDocument()
    expect(screen.getByText('87.535 ms')).toBeInTheDocument()
    expect(screen.getByText(/"wire_tensor_body_bytes"/)).toBeInTheDocument()
    expect(screen.queryByText('最终选择')).not.toBeInTheDocument()
  })
})

it('does not reuse the prior executed mode as the current adaptive request', async () => {
  serve(adaptiveSnapshot('KV'), [])
  await ready()
  act(() =>
    MockWebSocket.instances[0].emit({
      ...tokenSelectionEvent,
      type: 'REAL_ADAPTIVE_HANDOFF_STARTED',
      metadata: {},
    }),
  )
  expect(screen.getByText('本次请求 尚未选择 · 待完成')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: '通信运行中…' })).toBeDisabled()
})
