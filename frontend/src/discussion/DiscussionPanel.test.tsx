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
import DiscussionPanel from './DiscussionPanel'
import {
  getLatestDiscussion,
  switchModel,
  getDiscussion,
  previewDiscussionPrompts,
  startDiscussion,
  patchDiscussionNetwork,
} from './api'
import type {
  DiscussionSession,
  CostProfile,
  Handoff,
  LatestDiscussion,
  PromptConfig,
} from './types'
vi.mock('../api/dashboard', () => ({
  getRecentEvents: vi.fn(async () => ({ events: [], count: 0 })),
  eventWebSocketUrl: () => 'ws://localhost/ws/events',
}))
vi.mock('./api', () => ({
  getLatestDiscussion: vi.fn(),
  switchModel: vi.fn(),
  getDiscussion: vi.fn(),
  startDiscussion: vi.fn(),
  patchDiscussionNetwork: vi.fn(),
  previewDiscussionPrompts: vi.fn(),
}))
class Socket {
  static instances: Socket[] = []
  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  onmessage: ((e: MessageEvent) => void) | null = null
  constructor() {
    Socket.instances.push(this)
  }
  close() {}
  emit(type: string, metadata: Record<string, unknown>) {
    this.onmessage?.({
      data: JSON.stringify({
        type,
        timestamp: '2026-10-10T10:00:00Z',
        metadata,
      }),
    } as MessageEvent)
  }
}
const handoff: Handoff = {
  sequence: 1,
  message_id: 'A:1',
  source: 'A',
  target: 'B',
  discussion_round: 1,
  requested_mode: 'kv',
  executed_mode: 'kv',
  requested_kv_strategy: 'rope_aligned_delta',
  executed_kv_strategy: 'rope_aligned_delta',
  message_content_token_count: 15,
  wire_bytes: 187641,
  sender_kv_span: [80, 95],
  receiver_kv_target_span: [70, 85],
  receiver_cache_length_before: 63,
  receiver_cache_length_after: 87,
  receiver_message_prefill_count: 0,
  kv_repair_method: 'post_rope_rephase',
  rope_position_delta: -10,
  error: null,
  fallback_reason: null,
  append_succeeded: true,
  unpack_succeeded: true,
  cache_ledger_consistent: true,
}
const defaults: PromptConfig = {
  agent_a: { text: 'Server default A', source: 'default' },
  agent_b: { text: 'Server default B', source: 'default' },
  version: 'general-v1',
  stages: {
    a_first: 'A initial',
    b_first: 'B review',
    a_followup: 'A update',
    b_followup: 'B update review',
    final: 'Final using history',
  },
}
function session(
  overrides: Partial<DiscussionSession> = {},
): DiscussionSession {
  return {
    session_id: 'test-session',
    status: 'completed',
    revision: 19,
    phase: 'DISCUSSION_COMPLETED',
    round: 2,
    agent: 'A',
    sequence: 5,
    prompt_config: defaults,
    model: { model_name: 'session-model', device: 'mps', dtype: 'bfloat16' },
    config: {
      question: '23 + 48 = ?',
      rounds: 2,
      max_new_tokens: 64,
      summarizer: 'A',
      mode: 'kv:v1',
      handoff_plan: [],
    },
    turns: [
      {
        sequence: 1,
        round: 1,
        agent: 'A',
        text: 'opaque A output',
        token_count: 16,
        termination_reason: 'eos',
        message_id: 'A:1',
      },
      {
        sequence: 2,
        round: 1,
        agent: 'B',
        text: 'opaque B review',
        token_count: 8,
        termination_reason: 'eos',
        message_id: 'B:1',
      },
    ],
    handoffs: [handoff],
    final_answer: {
      agent: 'A',
      text: 'opaque final result',
      token_count: 9,
      termination_reason: 'eos',
      message_id: 'A:local:1',
    },
    summary: {
      rounds: 2,
      completed_turns: 4,
      handoff_count: 4,
      attempted_handoff_count: 4,
      token_handoffs: 0,
      kv_v0_handoffs: 0,
      kv_v1_handoffs: 4,
      total_wire_bytes: 410050,
      final_cache_lengths: { A: 220, B: 185 },
    },
    error: null,
    failure_stage: null,
    termination_reason: 'rounds_complete',
    cache_lengths: { A: 220, B: 185 },
    cache_released: true,
    transport: 'InMemoryByteTransport',
    created_at: '2026-10-10T09:00:00Z',
    updated_at: '2026-10-10T10:00:00Z',
    ...overrides,
  }
}
let latest: LatestDiscussion
beforeEach(() => {
  window.history.replaceState(null, '', '/')
  latest = {
    available: true,
    session: null,
    model: { model_name: 'mock-model', device: 'cpu', dtype: 'float32' },
  }
  vi.mocked(previewDiscussionPrompts).mockImplementation(async (a, b) => ({
    ...defaults,
    agent_a: a.trim() ? { text: a, source: 'custom' } : defaults.agent_a,
    agent_b: b.trim() ? { text: b, source: 'custom' } : defaults.agent_b,
  }))
  Socket.instances = []
  vi.stubGlobal('WebSocket', Socket)
  vi.mocked(getLatestDiscussion).mockImplementation(async () => latest)
  vi.mocked(startDiscussion).mockImplementation(async (config) => {
    latest.session = session({
      config,
      status: 'running',
      phase: 'DISCUSSION_STARTED',
      revision: 1,
      turns: [],
      handoffs: [],
      final_answer: null,
      summary: null,
    })
    return latest.session
  })
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
  vi.useRealTimers()
  window.history.replaceState(null, '', '/')
})
async function ready() {
  render(<DiscussionPanel />)
  if (latest.session) await screen.findByText('opaque A output')
  else
    await waitFor(() =>
      expect(screen.getByRole('button', { name: '开始讨论' })).toBeEnabled(),
    )
}
const dialog = () => within(screen.getByRole('dialog', { name: '实验配置' }))
const configure = (tab = '任务') => {
  if (!screen.queryByRole('dialog'))
    fireEvent.click(screen.getByRole('button', { name: '配置' }))
  if (tab !== '任务')
    fireEvent.click(dialog().getByRole('button', { name: tab }))
  else if (dialog().queryByRole('button', { name: '任务' }))
    fireEvent.click(dialog().getByRole('button', { name: '任务' }))
}
const change = (name: string, value: string) =>
  fireEvent.change(screen.getByLabelText(name), { target: { value } })
const start = () =>
  fireEvent.click(dialog().getByRole('button', { name: /^开始讨论$/ }))
const detail = (tab = '观测') => {
  fireEvent.click(screen.getByRole('button', { name: '实验详情' }))
  if (tab !== '观测') {
    const b = within(
      screen.getByRole('dialog', { name: '实验详情' }),
    ).getByRole('button', { name: tab })
    fireEvent.click(b)
  }
}
const closeConfig = () =>
  fireEvent.click(dialog().getByRole('button', { name: '关闭实验配置' }))

describe('Reading-first discussion controls', () => {
  it('places mini inside the communication status and full inside network details', async () => {
    latest.session = session()
    await ready()
    expect(within(screen.getByRole('region', { name: '双 Agent 通信状态' })).getByRole('region', { name: '最近通信模式切换' })).toBeInTheDocument()
    detail('网络轨迹')
    const full = within(screen.getByRole('dialog', { name: '实验详情' })).getByRole('region', { name: '通信模式切换时间线' })
    expect(within(full).getByRole('button', { name: /交接 #1/ })).toBeInTheDocument()
  })
  it('starts from the reading workspace and retains all draft fields', async () => {
    await ready()
    configure()
    change('用户问题', '12 * 13 = ?')
    change('讨论轮数', '3')
    change('最终总结 Agent', 'B')
    change('每次发言最大 Tokens', '32')
    start()
    await waitFor(() =>
      expect(startDiscussion).toHaveBeenCalledWith(
        {
          architecture: 'local', max_context_tokens: 8192, advanced_rounds: false, network_mode: 'shannon', channel: { bandwidth_mhz:20,snr_mean_db:10,snr_sigma_db:4,correlation:.85,seed:42,efficiency:.8 }, base_latency_ms: 0, bandwidth_trace: [],
          question: '12 * 13 = ?',
          rounds: 3,
          summarizer: 'B',
          max_new_tokens: 32,
          mode: 'auto', auto_policy: 'performance_experiment', auto_kv_version: 'v0', auto_candidates: ['token', 'kv:v0'], idealized_simulation: true, allow_experimental_v3: false, cost_profile: null, synthetic_assumptions: { schema: 'synthetic_compute_v1', source: 'synthetic_assumption', units: 'ms', a: 20, b: 2, c: .002, d: 5, e: .05, f: .2 },
          handoff_plan: [],
          agent_a_system_prompt: '',
          agent_b_system_prompt: '',
        },
        expect.any(String),
      ),
    )
  })
  it.each(['token', 'kv:v0', 'kv:v1', 'kv:v2', 'auto'])('submits uniform %s', async (mode) => {
    await ready()
    configure('通信')
    change('通信模式', mode)
    start()
    await waitFor(() =>
      expect(startDiscussion).toHaveBeenCalledWith(
        expect.objectContaining({ mode, handoff_plan: [] }),
        expect.any(String),
      ),
    )
  })
  it('preserves Mixed plan entries while changing rounds and submits strict direction order', async () => {
    await ready()
    configure('通信')
    change('通信模式', 'mixed')
    expect(screen.getAllByLabelText(/Handoff/)).toHaveLength(8)
    change('Handoff 1', 'kv:v0')
    configure()
    change('讨论轮数', '1')
    configure('通信')
    expect(screen.getAllByLabelText(/Handoff/)).toHaveLength(2)
    start()
    await waitFor(() =>
      expect(startDiscussion).toHaveBeenCalledWith(
        expect.objectContaining({
          handoff_plan: [
            { round: 1, source: 'A', target: 'B', mode: 'kv:v0' },
            { round: 1, source: 'B', target: 'A', mode: 'kv:v1' },
          ],
        }),
        expect.any(String),
      ),
    )
  })
  it('submits V2 ratio and rejects invalid tail values', async () => {
    await ready()
    configure('通信')
    change('通信模式', 'kv:v2')
    change('V2 重算策略', 'ratio')
    change('V2 重算值', '2')
    for (const button of screen.getAllByRole('button', { name: /开始/ })) expect(button).toBeDisabled()
    change('V2 重算值', '0.25')
    start()
    await waitFor(() => expect(startDiscussion).toHaveBeenCalledWith(
      expect.objectContaining({ mode: 'kv:v2', kv_recompute_ratio: 0.25 }), expect.any(String)))
  })
  it('submits fixed V3 without V2 recompute parameters', async () => {
    await ready()
    configure('通信')
    change('通信模式', 'kv:v3')
    start()
    await waitFor(() => expect(startDiscussion).toHaveBeenCalledWith(
      expect.objectContaining({ mode: 'kv:v3' }), expect.any(String)))
    const config = vi.mocked(startDiscussion).mock.calls[0][0]
    expect(config.kv_recompute_tokens).toBeUndefined()
    expect(config.kv_recompute_ratio).toBeUndefined()
  })
  it('enables experimental V2/V3 and fixed performance AUTO', async () => {
    await ready()
    configure('通信')
    expect(within(screen.getByLabelText('通信模式')).getByRole('option', { name: /V2/ })).not.toBeDisabled()
    expect(within(screen.getByLabelText('通信模式')).getByRole('option', { name: /V3/ })).not.toBeDisabled()
    expect(screen.getByRole('option', { name: /AUTO/ })).toBeEnabled()
  })
  it.each(['', '   '])('rejects empty question %j', async (q) => {
    await ready()
    configure()
    change('用户问题', q)
    expect(dialog().getByRole('button', { name: '开始讨论' })).toBeDisabled()
  })
  it.each(['0', '-1', '2.5'])('rejects token budget %s', async (n) => {
    await ready()
    configure()
    change('每次发言最大 Tokens', n)
    expect(dialog().getByRole('button', { name: '开始讨论' })).toBeDisabled()
  })
  it('locks the running experiment, rejects repeats and shows frozen configuration', async () => {
    await ready()
    const button = screen.getByRole('button', { name: '开始讨论' })
    fireEvent.click(button)
    fireEvent.click(button)
    await screen.findByRole('button', { name: '讨论运行中…' })
    expect(startDiscussion).toHaveBeenCalledTimes(1)
    configure()
    expect(screen.getByText(/模拟网络可独立调节/)).toBeInTheDocument()
    expect(dialog().queryByRole('textbox')).not.toBeInTheDocument()
    expect(dialog().getByText(/"mode": "auto"/)).toBeInTheDocument()
  })
  it('loads actual discussion, final answer, caches and full receipt on demand', async () => {
    latest.session = session()
    await ready()
    expect(screen.getByText('opaque B review')).toBeInTheDocument()
    expect(screen.queryByText('opaque final result')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '最终答案' }))
    expect(screen.getByText('opaque final result')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '讨论' }))
    fireEvent.click(screen.getByRole('button', { name: 'B 已接收 KV V1' }))
    expect(
      screen.getByText('接收方消息预填充次数').nextElementSibling,
    ).toHaveTextContent('0')
    fireEvent.click(screen.getByRole('button', { name: '观测' }))
    expect(
      screen.getByText('总 wire bytes').nextElementSibling,
    ).toHaveTextContent('410,050 B')
    expect(
      screen.getByText('Agent A 最终 Cache Tokens').nextElementSibling,
    ).toHaveTextContent('220')
    expect(startDiscussion).not.toHaveBeenCalled()
  })
  it('shows failed handoff separately from a confirmed reception', async () => {
    latest.session = session({
      status: 'failed',
      phase: 'DISCUSSION_FAILED',
      error: 'append failed',
      failure_stage: 'handoff',
      final_answer: null,
      handoffs: [
        {
          ...handoff,
          executed_mode: null,
          executed_kv_strategy: null,
          append_succeeded: false,
          error: 'append failed',
        },
      ],
    })
    await ready()
    expect(screen.getByRole('alert')).toHaveTextContent('append failed')
    expect(
      screen.getByRole('button', { name: '交接失败 · 未确认' }),
    ).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '最终答案' }))
    expect(screen.getByText('尚无最终模型结果')).toBeInTheDocument()
  })
  it('shows empty truncated model answers truthfully', async () => {
    latest.session = session({
      final_answer: {
        agent: 'B',
        text: '',
        token_count: 2,
        termination_reason: 'max_new_tokens',
        message_id: 'B:local:1',
      },
    })
    await ready()
    fireEvent.click(screen.getByRole('button', { name: '最终答案' }))
    expect(screen.getByText('（模型生成内容为空）')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '实验详情' }))
    fireEvent.click(screen.getByRole('button', { name: '消息生成' }))
    expect(screen.getByText('达到生成上限，内容可能未完成')).toBeInTheDocument()
  })
  it('keeps model-unavailable and REST errors actionable', async () => {
    latest.available = false
    render(<DiscussionPanel />)
    expect(await screen.findByRole('alert')).toHaveTextContent('模型未加载')
    expect(screen.getByRole('button', { name: '开始讨论' })).toBeDisabled()
  })
  it('preserves draft on a REST rejection', async () => {
    vi.mocked(startDiscussion).mockRejectedValue(
      new Error('One Session is already running'),
    )
    await ready()
    configure()
    start()
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'already running',
    )
    expect(screen.getByLabelText('用户问题')).toHaveValue('23 + 48 = ?')
  })
  it('does not claim delivery at HANDOFF_STARTED and updates from actual trace', async () => {
    latest.session = session({
      revision: 3,
      status: 'running',
      phase: 'AGENT_TURN_COMPLETED',
      handoffs: [],
      final_answer: null,
      summary: null,
    })
    await ready()
    act(() =>
      Socket.instances[0].emit('HANDOFF_STARTED', {
        session_id: 'test-session',
        revision: 4,
        status: 'running',
        round: 1,
        agent: 'A',
        sequence: 1,
      }),
    )
    expect(screen.getAllByText(/尚未确认交付/).length).toBeGreaterThan(0)
    expect(screen.getAllByText('已生成 · 尚无关联交付记录')[0]).toBeInTheDocument()
    act(() =>
      Socket.instances[0].emit('HANDOFF_COMPLETED', {
        session_id: 'test-session',
        revision: 5,
        status: 'running',
        round: 1,
        agent: 'A',
        sequence: 1,
        handoff,
      }),
    )
    expect(
      screen.getByRole('button', { name: 'B 已接收 KV V1' }),
    ).toBeInTheDocument()
  })
  it('recovers WS reconnect through GET without starting inference', async () => {
    latest.session = session({
      status: 'running',
      phase: 'DISCUSSION_STARTED',
      revision: 1,
      turns: session().turns,
      handoffs: [],
      final_answer: null,
      summary: null,
    })
    await ready()
    act(() => Socket.instances[0].onopen?.())
    latest.session = session()
    vi.useFakeTimers()
    act(() => Socket.instances[0].onclose?.())
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    act(() => Socket.instances[1].onopen?.())
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })
    fireEvent.click(screen.getByRole('button', { name: '最终答案' }))
    expect(screen.getByText('opaque final result')).toBeInTheDocument()
    expect(startDiscussion).not.toHaveBeenCalled()
  })
  it('repairs a revision gap using authoritative REST instead of accepting incomplete evidence', async () => {
    latest.session = session({
      revision: 3,
      status: 'running',
      handoffs: [],
      final_answer: null,
      summary: null,
    })
    await ready()
    latest.session = session({ revision: 9 })
    act(() =>
      Socket.instances[0].emit('HANDOFF_COMPLETED', {
        session_id: 'test-session',
        revision: 9,
        status: 'running',
        handoff,
      }),
    )
    await screen.findByRole('button', { name: 'B 已接收 KV V1' })
    expect(getLatestDiscussion).toHaveBeenCalledTimes(2)
    expect(startDiscussion).not.toHaveBeenCalled()
  })
  it('keeps reading positions per tab and session while evidence opens', async () => {
    latest.session = session()
    await ready()
    const reader = screen.getByRole('region', { name: '实验内容阅读区' })
    reader.scrollTop = 177
    fireEvent.scroll(reader)
    fireEvent.click(screen.getByRole('button', { name: '最终答案' }))
    expect(reader.scrollTop).toBe(0)
    fireEvent.click(screen.getByRole('button', { name: '讨论' }))
    expect(reader.scrollTop).toBe(177)
    fireEvent.click(screen.getByRole('button', { name: 'B 已接收 KV V1' }))
    fireEvent.click(screen.getByRole('button', { name: '关闭实验详情' }))
    expect(reader.scrollTop).toBe(177)
  })
})

describe('Independent Agent prompts', () => {
  it('starts with independent empty overrides and server defaults in preview', async () => {
    await ready()
    configure('Agent')
    expect(screen.getByLabelText('Agent A System Prompt')).toHaveValue('')
    expect(screen.getByLabelText('Agent B System Prompt')).toHaveValue('')
    configure('生效预览')
    expect(screen.getByText('Server default A')).toBeInTheDocument()
    expect(screen.getByText('Server default B')).toBeInTheDocument()
    expect(screen.getByText('B 检查阶段指令')).toBeInTheDocument()
  })
  it.each(['A', 'B'])(
    'previews and submits Agent %s independently',
    async (agent) => {
      await ready()
      configure('Agent')
      change(`Agent ${agent} System Prompt`, `Custom ${agent}`)
      await waitFor(() =>
        expect(
          dialog().getByRole('button', { name: '开始讨论' }),
        ).toBeEnabled(),
      )
      configure('生效预览')
      expect(
        screen.getByText(`Custom ${agent}`, { selector: 'pre' }),
      ).toBeInTheDocument()
      start()
      await waitFor(() =>
        expect(startDiscussion).toHaveBeenCalledWith(
          expect.objectContaining({
            [agent === 'A' ? 'agent_a_system_prompt' : 'agent_b_system_prompt']:
              `Custom ${agent}`,
            [agent === 'A' ? 'agent_b_system_prompt' : 'agent_a_system_prompt']:
              '',
          }),
          expect.any(String),
        ),
      )
    },
  )
  it('treats whitespace as default and retains exact draft', async () => {
    await ready()
    configure('Agent')
    change('Agent A System Prompt', ' \n')
    change('Agent B System Prompt', 'Custom B')
    await waitFor(() =>
      expect(dialog().getByRole('button', { name: '开始讨论' })).toBeEnabled(),
    )
    configure('生效预览')
    expect(screen.getByText('Server default A')).toBeInTheDocument()
    start()
    await waitFor(() =>
      expect(startDiscussion).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_a_system_prompt: ' \n',
          agent_b_system_prompt: 'Custom B',
        }),
        expect.any(String),
      ),
    )
  })
  it('resets one Agent without clearing the other and resets both', async () => {
    await ready()
    configure('Agent')
    change('Agent A System Prompt', 'Custom A')
    change('Agent B System Prompt', 'Custom B')
    fireEvent.click(screen.getByRole('button', { name: '恢复 Agent A 默认' }))
    expect(screen.getByLabelText('Agent A System Prompt')).toHaveValue('')
    expect(screen.getByLabelText('Agent B System Prompt')).toHaveValue(
      'Custom B',
    )
    fireEvent.click(screen.getByRole('button', { name: '恢复 Agent B 默认' }))
    expect(screen.getByLabelText('Agent B System Prompt')).toHaveValue('')
    change('Agent A System Prompt', 'Another A')
    fireEvent.click(screen.getByRole('button', { name: '恢复默认' }))
    expect(screen.getByLabelText('Agent A System Prompt')).toHaveValue('')
  })
  it('separates immutable session prompts from next-session draft', async () => {
    latest.session = session({
      config: { ...session().config, agent_b_system_prompt: 'Saved B' },
      prompt_config: {
        ...defaults,
        agent_b: { text: 'Saved B', source: 'custom' },
      },
    })
    await ready()
    configure('Agent')
    expect(screen.getByLabelText('Agent B System Prompt')).toHaveValue(
      'Saved B',
    )
    change('Agent B System Prompt', 'Next B')
    closeConfig()
    detail('本次配置')
    expect(screen.getByText('Saved B', { selector: 'pre' })).toBeInTheDocument()
    expect(screen.getByText(/session-model · mps/)).toBeInTheDocument()
    expect(startDiscussion).not.toHaveBeenCalled()
  })
  it('blocks invalid prompts and recovers after reset', async () => {
    await ready()
    configure('Agent')
    vi.mocked(previewDiscussionPrompts).mockRejectedValueOnce(
      new Error('System Prompt must not contain chat control tokens'),
    )
    change('Agent B System Prompt', '<s>bad')
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'chat control tokens',
    )
    expect(dialog().getByRole('button', { name: '开始讨论' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '恢复 Agent B 默认' }))
    await waitFor(() =>
      expect(dialog().getByRole('button', { name: '开始讨论' })).toBeEnabled(),
    )
  })
  it('ignores a stale asynchronous prompt preview', async () => {
    await ready()
    configure('Agent')
    let resolveOld!: (value: PromptConfig) => void
    vi.mocked(previewDiscussionPrompts).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveOld = resolve
        }),
    )
    change('Agent B System Prompt', 'Old B')
    await waitFor(() => expect(resolveOld).toBeDefined())
    change('Agent B System Prompt', 'New B')
    await waitFor(() =>
      expect(dialog().getByRole('button', { name: '开始讨论' })).toBeEnabled(),
    )
    await act(async () =>
      resolveOld({ ...defaults, agent_b: { text: 'Old B', source: 'custom' } }),
    )
    configure('生效预览')
    expect(screen.queryByText('Old B')).not.toBeInTheDocument()
    expect(screen.getByText('New B', { selector: 'pre' })).toBeInTheDocument()
  })
})

describe('Private number collaboration', () => {
  it.each(['token', 'kv:v0', 'kv:v1', 'mixed'])(
    'submits isolated scene with %s and fixed round/summarizer',
    async (mode) => {
      await ready()
      configure()
      change('任务模式', 'secret_sum')
      change('Agent B 的秘密数字', '31')
      expect(screen.queryByLabelText('用户问题')).not.toBeInTheDocument()
      expect(screen.getByLabelText('最终总结 Agent')).toBeDisabled()
      configure('通信')
      change('通信模式', mode)
      if (mode === 'mixed')
        expect(screen.getAllByLabelText(/Handoff/)).toHaveLength(2)
      fireEvent.click(dialog().getByRole('button', { name: '开始协作' }))
      await waitFor(() =>
        expect(startDiscussion).toHaveBeenCalledWith(
          expect.objectContaining({
            scenario: 'secret_sum',
            secret_a: 17,
            secret_b: 31,
            rounds: 1,
            summarizer: 'A',
            mode,
          }),
          expect.any(String),
        ),
      )
      expect(
        vi.mocked(startDiscussion).mock.calls[0][0].agent_a_system_prompt,
      ).toBeUndefined()
    },
  )
  it.each(['0', '100', '2.5', ''])('rejects invalid secret %j', async (n) => {
    await ready()
    configure()
    change('任务模式', 'secret_sum')
    change('Agent A 的秘密数字', n)
    expect(dialog().getByRole('button', { name: '开始协作' })).toBeDisabled()
    expect(startDiscussion).not.toHaveBeenCalled()
  })
  it('restores private configuration and reports quality separately from engineering', async () => {
    latest.session = session({
      config: {
        scenario: 'secret_sum',
        secret_a: 17,
        secret_b: 31,
        question: 'Public task',
        rounds: 1,
        summarizer: 'A',
        mode: 'token',
        max_new_tokens: 96,
        handoff_plan: [],
      },
      score: {
        engineering_success: true,
        a_sent_own_secret: true,
        b_sent_own_secret: true,
        b_received_a: null,
        a_received_b: false,
        information_exchange_success: false,
        final_sum_correct: true,
        task_success: true,
        expected_sum: 48,
        note: 'posthoc',
      },
    })
    await ready()
    configure()
    expect(screen.getByLabelText('Agent B 的秘密数字')).toHaveValue(31)
    closeConfig()
    detail()
    expect(
      screen.getByText('工程执行成功').nextElementSibling,
    ).toHaveTextContent('是')
    expect(
      screen.getByText('B 正确使用 A 的秘密').nextElementSibling,
    ).toHaveTextContent('无法判定')
    expect(
      screen.getByText('信息交换成功').nextElementSibling,
    ).toHaveTextContent('否')
    expect(
      screen.getByText('最终答案正确').nextElementSibling,
    ).toHaveTextContent('是')
    expect(startDiscussion).not.toHaveBeenCalled()
  })
  it('retains ordinary discussion draft after switching scenes', async () => {
    await ready()
    configure()
    change('用户问题', 'Custom question')
    change('任务模式', 'secret_sum')
    change('任务模式', 'discussion')
    expect(screen.getByLabelText('用户问题')).toHaveValue('Custom question')
    start()
    await waitFor(() =>
      expect(startDiscussion).toHaveBeenCalledWith(
        expect.objectContaining({ question: 'Custom question', rounds: 4 }),
        expect.any(String),
      ),
    )
    expect(vi.mocked(startDiscussion).mock.calls[0][0].scenario).toBeUndefined()
  })
  it('projects posthoc score from completion events', async () => {
    latest.session = session({ status: 'running', revision: 10, score: null })
    await ready()
    act(() =>
      Socket.instances[0].emit('DISCUSSION_COMPLETED', {
        session_id: 'test-session',
        revision: 11,
        status: 'completed',
        round: 1,
        agent: 'A',
        sequence: 3,
        score: {
          engineering_success: true,
          a_sent_own_secret: true,
          b_sent_own_secret: true,
          b_received_a: true,
          a_received_b: null,
          information_exchange_success: null,
          final_sum_correct: false,
          task_success: false,
          expected_sum: 43,
        },
      }),
    )
    detail()
    expect(
      screen.getByText('最终答案正确').nextElementSibling,
    ).toHaveTextContent('否')
    expect(
      screen.getByText('A 正确使用 B 的秘密').nextElementSibling,
    ).toHaveTextContent('无法判定')
  })
})

describe('Ambiguous startup and process reset recovery', () => {
  it('recovers an accepted POST with a lost response without starting twice', async () => {
    await ready()
    vi.mocked(startDiscussion).mockImplementationOnce(async (config, id) => {
      latest.session = session({ session_id: id, config })
      throw new Error('response lost')
    })
    fireEvent.click(screen.getByRole('button', { name: '开始讨论' }))
    await screen.findByText('opaque A output')
    await waitFor(() =>
      expect(screen.queryByRole('alert')).not.toBeInTheDocument(),
    )
    expect(startDiscussion).toHaveBeenCalledTimes(1)
  })
  it('explains a lost process-local session without automatically POSTing again', async () => {
    latest.session = session()
    await ready()
    latest.session = null
    act(() => Socket.instances[0].onopen?.())
    expect(
      await screen.findByText(/服务器当前没有此会话记录/),
    ).toBeInTheDocument()
    expect(screen.getByLabelText('实验问题')).toHaveValue('23 + 48 = ?')
    expect(startDiscussion).not.toHaveBeenCalled()
  })
})

const inlineProfile: CostProfile = { version: 'fixture-v1', provenance: 'synthetic/test_only', model_identity: { sender: { model_id: 'fixture' }, receiver: { model_id: 'fixture' } }, stages: { token_handoff: { fixed_ms: 100 }, kv_base: { fixed_ms: 1 }, kv_contract_startup: { fixed_ms: 0 } }, sample_count: 0, message_token_range: [0, 128], cache_token_range: [0, 4096], notes: 'Synthetic component test only.' }
describe('Desktop AUTO configuration controls', () => {
  it('submits trace AUTO through the existing controller with numeric fields', async () => {
    await ready(); configure('通信'); change('通信模式','auto'); configure('网络'); change('网络模式','trace'); change('带宽轨迹 Mbps','20,1000,50,500'); change('单向基础延迟 ms','2'); start()
    await waitFor(() => expect(startDiscussion).toHaveBeenCalledWith(expect.objectContaining({ mode: 'auto', auto_policy: 'performance_experiment', idealized_simulation: true, cost_profile: null, network_mode: 'trace', bandwidth_trace: [20,1000,50,500], base_latency_ms: 2 }),expect.any(String)))
  })
  it('requires a profile, selects one KV and sends explicit V3 permission', async () => {
    await ready(); configure('通信'); change('通信模式','auto'); fireEvent.click(screen.getByLabelText('理想化场景仿真'))
    expect(dialog().getByRole('button',{name:'开始讨论'})).toBeDisabled()
    change('成本 Profile JSON',JSON.stringify(inlineProfile)); change('AUTO 的 KV 版本','v3')
    expect(dialog().getByRole('button',{name:'开始讨论'})).toBeDisabled(); fireEvent.click(screen.getByLabelText('额外许可 KV V3'))
    start(); await waitFor(() => expect(startDiscussion).toHaveBeenCalledWith(expect.objectContaining({ cost_profile: inlineProfile, allow_experimental_v3: true, auto_candidates: ['token','kv:v3'] }),expect.any(String)))
  })
  it('shows synthetic source even while viewing the network tab', async () => {
    await ready(); configure('通信'); change('通信模式','auto'); fireEvent.click(screen.getByLabelText('理想化场景仿真')); change('成本 Profile JSON',JSON.stringify(inlineProfile)); configure('网络')
    expect(dialog().getByText(/Synthetic \/ Test Only/)).toBeInTheDocument()
    change('网络模式','trace'); change('带宽轨迹 Mbps','20,,50'); expect(dialog().getByRole('button',{name:'开始讨论'})).toBeDisabled(); expect(dialog().getByRole('alert')).toHaveTextContent('轨迹')
  })
  it('imports browser JSON content without sending any path', async () => {
    await ready(); configure('通信'); change('通信模式','auto'); fireEvent.click(screen.getByLabelText('理想化场景仿真'))
    const file = new File(['fixture'],'profile.json',{type:'application/json'}); Object.defineProperty(file,'text',{value: async () => JSON.stringify(inlineProfile)})
    fireEvent.change(screen.getByLabelText('导入本地 Profile JSON'),{target:{files:[file]}})
    await waitFor(() => expect(screen.getByLabelText('成本 Profile JSON')).toHaveValue(JSON.stringify(inlineProfile)))
    start(); await waitFor(() => expect(startDiscussion).toHaveBeenCalledWith(expect.objectContaining({ cost_profile: inlineProfile }),expect.any(String)))
  })
  it('keeps final summaries free of inherited TOKEN/KV delivery badges', async () => {
    latest.session = session(); await ready(); fireEvent.click(screen.getByRole('button',{name:'最终答案'}))
    fireEvent.click(screen.getByRole('button', { name: '实验详情' }))
    fireEvent.click(screen.getByRole('button', { name: '消息生成' }))
    expect(screen.getByText(/本地总结，无额外通信交付/)).toBeInTheDocument()
    expect(within(screen.getByRole('region',{name:'实验内容阅读区'})).queryByRole('button',{name:/已接收/})).not.toBeInTheDocument()
  })
})


it('opens an older AUTO session through REST and preserves its URL without restarting', async () => {
  const older = session({ session_id: 'older-auto', created_at: '2026-10-10T08:00:00Z', config: { question: 'Historical AUTO', rounds: 1, max_new_tokens: 16, summarizer: 'A', mode: 'auto', handoff_plan: [] } })
  latest.session = session()
  latest.recent_sessions = [{ session_id: 'test-session', status: 'completed', question: 'Current', mode: 'kv:v1', created_at: latest.session.created_at }, { session_id: older.session_id, status: 'completed', question: 'Historical AUTO', mode: 'auto', created_at: older.created_at }]
  vi.mocked(getDiscussion).mockResolvedValue(older)
  await ready(); detail('会话'); fireEvent.click(screen.getByRole('button',{name:'打开会话'}))
  await waitFor(() => expect(getDiscussion).toHaveBeenCalledWith('older-auto'))
  await waitFor(() => expect(screen.getByRole('heading',{level:1})).toHaveTextContent('Historical AUTO'))
  expect(window.location.search).toBe('?discussion=older-auto'); expect(startDiscussion).not.toHaveBeenCalled()
})

it('restores a selected history URL through GET while another run prevents duplicate starts', async () => {
  const older = session({session_id:'older-url'})
  latest.session = session({status:'running'}); latest.active_session_id = 'test-session'
  vi.mocked(getDiscussion).mockResolvedValue(older)
  window.history.replaceState(null, '', '/?discussion=older-url')
  await ready(); expect(getDiscussion).toHaveBeenCalledWith('older-url'); expect(screen.getByRole('button',{name:'讨论运行中…'})).toBeDisabled(); expect(startDiscussion).not.toHaveBeenCalled()
})

it('recovers latest after a history record is evicted or a process restarts', async () => {
  latest.session = session(); vi.mocked(getDiscussion).mockRejectedValue(new Error('Discussion Session not found (process-local history)'))
  window.history.replaceState(null, '', '/?discussion=evicted')
  await ready(); expect(window.location.search).toBe(''); expect(startDiscussion).not.toHaveBeenCalled()
})


it('unlocks a completed session when WS terminal arrives after an active REST snapshot', async () => {
  latest.session = session({ status: 'running', revision: 18, phase: 'HANDOFF_COMPLETED' }); latest.active_session_id = 'test-session'
  await ready(); expect(screen.getByRole('button',{name:'讨论运行中…'})).toBeDisabled()
  act(() => Socket.instances[0].emit('DISCUSSION_COMPLETED',{session_id:'test-session',revision:19,status:'completed',sequence:5,round:2,agent:'A'}))
  await screen.findByRole('button',{name:'新实验'}); expect(screen.getByRole('button',{name:'新实验'})).toBeEnabled(); expect(startDiscussion).not.toHaveBeenCalled()
})


it('keeps synthetic source visible in secret_sum network configuration too', async () => {
  await ready(); configure(); change('任务模式','secret_sum'); configure('通信'); change('通信模式','auto'); fireEvent.click(screen.getByLabelText('理想化场景仿真')); change('成本 Profile JSON',JSON.stringify(inlineProfile)); configure('网络')
  expect(dialog().getByText(/Synthetic \/ Test Only/)).toBeInTheDocument(); expect(dialog().getByText(/一轮私有数字协作/)).toBeInTheDocument()
})

function enableDebug() {
  fireEvent.click(screen.getByRole('button', { name: '实验详情' }))
  fireEvent.click(screen.getByRole('radio', { name: '调试' }))
  fireEvent.click(screen.getByRole('button', { name: '关闭实验详情' }))
}

it('defaults to clean presentation and switches diagnostics without submitting or losing text', async () => {
  const base = session()
  latest.session = session({ config: { ...base.config, mode: 'auto', cost_profile: inlineProfile },
    turns: base.turns.map(t => ({ ...t, termination_reason: 'max_new_tokens' })) })
  await ready()
  expect(screen.getByText('· 模拟成本 · 测试')).toBeInTheDocument()
  expect(document.querySelector('.profile-source')).toBeNull()
  expect(document.querySelector('.network-status')).toBeNull()
  expect(document.querySelector('.generation-meta')).toBeNull()
  expect(document.querySelector('.channel-repair')).toBeNull()
  expect(screen.getByRole('button', { name: 'B 已接收 KV V1' })).toBeInTheDocument()
  const reader = screen.getByRole('region', { name: '实验内容阅读区' })
  reader.scrollTop = 40
  const originalReader = reader.innerHTML
  enableDebug()
  expect(document.querySelector('.network-status')).toBeNull()
  expect(document.querySelector('.generation-meta')).toBeNull()
  expect(document.querySelector('.profile-source')).toBeNull()
  expect(document.querySelector('.channel-repair')).toBeNull()
  expect(reader.scrollTop).toBe(40)
  expect(screen.getByRole('button', { name: 'B 已接收 KV V1 · 本地交接 不可用' })).toBeInTheDocument()
  expect(reader.querySelectorAll('.turn-text')).toHaveLength(2)
  fireEvent.click(screen.getByRole('button', { name: '实验详情' }))
  fireEvent.click(screen.getByRole('radio', { name: '展示' }))
  fireEvent.click(screen.getByRole('button', { name: '关闭实验详情' }))
  expect(document.querySelector('.generation-meta')).toBeNull()
  expect(document.querySelector('.network-status')).toBeNull()
  expect(screen.getByText('opaque A output')).toBeInTheDocument()
  expect(reader.scrollTop).toBe(40)
  expect(reader.innerHTML).toBe(originalReader)
  expect(startDiscussion).not.toHaveBeenCalled()
})

it('keeps generation diagnostics and receipt evidence accessible in presentation details', async () => {
  latest.session = session()
  await ready()
  fireEvent.click(screen.getByRole('button', { name: '实验详情' }))
  const details = within(screen.getByRole('dialog', { name: '实验详情' }))
  expect(details.getByRole('radio', { name: '展示' })).toBeChecked()
  fireEvent.click(details.getByRole('button', { name: '消息生成' }))
  const generated = details.getByRole('region', { name: 'Agent A · 第 1 轮' })
  expect(within(generated).getByText('生成 Tokens').nextElementSibling).toHaveTextContent('16')
  expect(within(generated).getByText('正常结束（EOS）')).toBeInTheDocument()
  fireEvent.click(details.getByRole('button', { name: '通信证据' }))
  expect(details.getByText('接收方 Cache · 交付后')).toBeInTheDocument()
  expect(details.getByText('选中策略')).toBeInTheDocument()
  expect(details.getByText('实际交付')).toBeInTheDocument()
  expect(startDiscussion).not.toHaveBeenCalled()
})

it.each(['512', '1024', '8'])('accepts a normal or custom small generation budget %s', async tokens => {
  await ready()
  configure()
  fireEvent.change(dialog().getByRole('spinbutton', { name: '每次发言最大 Tokens' }), { target: { value: tokens } })
  expect(dialog().getByRole('button', { name: '开始讨论' })).toBeEnabled()
})

it('shows truncation only on affected messages and preserves the reading surface', async () => {
  const base = session()
  latest.session = session({ turns: [base.turns[0], { ...base.turns[1], termination_reason: 'context_limit', configured_max_new_tokens: 1024, effective_max_new_tokens: 256, context_remaining_tokens: 256, budget_limit_reason: 'context_limit' }] })
  await ready()
  const reader = screen.getByRole('region', { name: '实验内容阅读区' })
  expect(within(reader).getByText(/回答达到上下文预算限制/)).toBeInTheDocument()
  expect(within(reader).queryByText(/回答达到生成上限/)).not.toBeInTheDocument()
  expect(within(reader).queryByText(/1024|256 Tokens/)).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '实验详情' }))
  fireEvent.click(screen.getByRole('button', { name: '消息生成' }))
  expect(screen.getAllByText('生效预算 Tokens').length).toBeGreaterThan(0) // first legacy turn also reports unknown
})

it('updates a running trace via explicit manual override and restores server revision', async () => {
  const base = session({ status: 'running', phase: 'AGENT_TURN_STARTED', revision: 3, config: { ...session().config, network_mode: 'trace', bandwidth_trace: [20, 1000] } })
  base.network = { config: { mode: 'trace', bandwidth_mbps: 20, base_latency_ms: 0, trace_mbps: [20, 1000], source: 'rest_session_config' }, config_revision: 0, next_unfrozen_handoff_index: 2, is_simulated: true, effective_rule: 'next_unfrozen_handoff' }
  latest.session = base
  const updated = { ...base, revision: 4, network: { ...base.network, config: { ...base.network.config, mode: 'manual' as const, bandwidth_mbps: 50, base_latency_ms: 2, trace_mbps: [], source: 'rest_manual_override' }, config_revision: 1 } }
  vi.mocked(patchDiscussionNetwork).mockResolvedValue({ updated: true, revision: 4, config_revision: 1, effective_from_handoff_index: 2, before: base.network, after: updated.network, accepted_at: '2026-10-15T00:00:00Z', source: 'rest_manual_override', mode_before: 'trace', mode_after: 'manual' })
  vi.mocked(getDiscussion).mockResolvedValue(updated)
  await ready()
  configure()
  expect(screen.getByRole('button', { name: '应用模拟网络' })).toBeDisabled()
  fireEvent.change(screen.getByRole('spinbutton', { name: '活动模拟带宽 Mbps' }), { target: { value: '50' } })
  fireEvent.change(screen.getByRole('spinbutton', { name: '活动单向基础延迟 ms' }), { target: { value: '2' } })
  fireEvent.click(screen.getByRole('checkbox', { name: /切换到 manual override/ }))
  fireEvent.click(screen.getByRole('button', { name: '应用模拟网络' }))
  expect(await screen.findByText(/已更新模拟网络 · 版本 1/)).toBeInTheDocument()
  expect(patchDiscussionNetwork).toHaveBeenCalledWith(base.session_id, { bandwidth_mbps: 50, base_latency_ms: 2, manual_override: true })
  expect(screen.queryByRole('checkbox', { name: /切换到 manual override/ })).not.toBeInTheDocument()
  expect(startDiscussion).not.toHaveBeenCalled()
})


describe('Current backend model details', () => {
  it('shows the loaded snapshot rather than the model of a historical session', async () => {
    latest.session = session()
    latest.model = {
      model_name: '/cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/abc123',
      device: 'mps', dtype: 'torch.bfloat16',
    }
    render(<DiscussionPanel />)
    const trigger = await screen.findByRole('button', { name: '查看模型详情：Qwen2.5-1.5B-Instruct' })
    trigger.focus()
    fireEvent.click(trigger)
    const dialog = screen.getByRole('dialog', { name: '模型详情' })
    expect(within(dialog).getByText('Qwen/Qwen2.5-1.5B-Instruct')).toBeInTheDocument()
    expect(within(dialog).getByText('MPS')).toBeInTheDocument()
    expect(within(dialog).getByText('BF16')).toBeInTheDocument()
    expect(within(dialog).getByText('abc123')).toBeInTheDocument()
    expect(within(dialog).queryByText('session-model')).not.toBeInTheDocument()
    fireEvent.click(within(dialog).getByRole('button', { name: '返回页面' }))
    expect(screen.queryByRole('dialog', { name: '模型详情' })).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()
  })

  it('does not report a historical model as loaded when the backend is unavailable', async () => {
    latest.available = false
    latest.model = null
    latest.session = session()
    render(<DiscussionPanel />)
    await screen.findByText('模型未加载')
    expect(screen.queryByRole('button', { name: /查看模型详情/ })).not.toBeInTheDocument()
  })
})


describe('Local model switching', () => {
  const switching = () => {
    latest.model = { model_name: '/cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/abc', device: 'mps', dtype: 'bfloat16' }
    latest.model_switch = { loading: false, error: null, options: ['0.5B', '1.5B', '3B'] }
  }
  it('lists all models and marks current; Escape restores trigger focus', async () => {
    switching()
    render(<DiscussionPanel />)
    const trigger = await screen.findByRole('button', { name: /切换模型/ })
    fireEvent.click(trigger)
    expect(screen.getAllByRole('menuitemradio')).toHaveLength(3)
    expect(screen.getByRole('menuitemradio', { name: /1.5B/ })).toHaveAttribute('aria-checked', 'true')
    fireEvent.keyDown(trigger, { key: 'Escape' })
    expect(screen.queryByRole('menu')).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()
  })
  it('dispatches real API selection and displays loading', async () => {
    switching()
    vi.mocked(switchModel).mockResolvedValue({ ...latest, available: false, model_switch: { ...latest.model_switch!, loading: true } })
    render(<DiscussionPanel />)
    fireEvent.click(await screen.findByRole('button', { name: /切换模型/ }))
    act(() => Socket.instances[0].onopen?.())
    await waitFor(() => expect(screen.getByRole('menuitemradio', { name: '3B' })).toBeEnabled())
    fireEvent.click(screen.getByRole('menuitemradio', { name: '3B' }))
    expect(await screen.findByText('正在加载模型')).toBeInTheDocument()
    expect(switchModel).toHaveBeenCalledWith('3B')
    expect(startDiscussion).not.toHaveBeenCalled()
  })
  it('locks all options during discussion', async () => {
    switching()
    latest.session = session({ status: 'running' })
    render(<DiscussionPanel />)
    fireEvent.click(await screen.findByRole('button', { name: /切换模型/ }))
    for (const item of screen.getAllByRole('menuitemradio')) expect(item).toBeDisabled()
  })
  it('shows the actual failure while retaining the loaded model name', async () => {
    switching()
    vi.mocked(switchModel).mockRejectedValue(new Error('Local cache shard missing'))
    render(<DiscussionPanel />)
    fireEvent.click(await screen.findByRole('button', { name: /切换模型/ }))
    act(() => Socket.instances[0].onopen?.())
    await waitFor(() => expect(screen.getByRole('menuitemradio', { name: '3B' })).toBeEnabled())
    fireEvent.click(screen.getByRole('menuitemradio', { name: '3B' }))
    expect(await screen.findByText('Local cache shard missing')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /切换模型：Qwen2.5-1.5B/ })).toBeInTheDocument()
  })
})


it('debug time is associated by message ID and absent from presentation', async () => {
  latest.session = session({ handoffs: [{ ...handoff, observed_local_ms: 83.456 }, { ...handoff, sequence: 2, message_id: 'B:1', source: 'B', target: 'A', executed_mode: 'token', observed_local_ms: 21.123 }] })
  await ready()
  expect(screen.queryByText(/本地交接/)).not.toBeInTheDocument()
  enableDebug()
  expect(screen.getByRole('button', { name: 'B 已接收 KV V1 · 本地交接 83.46 ms' })).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'A 已接收 TOKEN · 本地交接 21.12 ms' })).toBeInTheDocument()
  expect(screen.getByText('opaque A output')).toBeInTheDocument()
  expect(startDiscussion).not.toHaveBeenCalled()
})

it('idealized switch defaults on and submits a separate source', async () => {
  await ready(); configure('通信'); change('通信模式','auto')
  const toggle = screen.getByLabelText('理想化场景仿真')
  expect(toggle).toBeChecked()
  change('AUTO 的 KV 版本','v1')
  expect(screen.queryByLabelText('成本 Profile JSON')).not.toBeInTheDocument()
  expect(screen.getByText(/当前候选：TOKEN ↔ KV V1/)).toBeInTheDocument()
  start()
  await waitFor(() => expect(startDiscussion).toHaveBeenCalledWith(expect.objectContaining({ auto_candidates: ['token','kv:v1'], auto_kv_version: 'v1', idealized_simulation: true, cost_profile: null }), expect.any(String)))
})

it('removes architecture selection and preserves the Local model menu', async () => {
  latest.model_switch={loading:false,error:null,options:['0.5B','1.5B','3B']}
  await ready()
  expect(screen.queryByLabelText('运行架构')).not.toBeInTheDocument()
  expect(screen.getByRole('button',{name:/切换模型/})).toBeInTheDocument()
  expect(switchModel).not.toHaveBeenCalled()
})
it('keeps the architecture selector absent during an active Local discussion',async()=>{
  latest.session=session({status:'running',phase:'AGENT_TURN_STARTED'})
  await ready()
  expect(screen.queryByLabelText('运行架构')).not.toBeInTheDocument()
})
it('a distributed receipt with unknown outcome cannot show the received label',async()=>{
  window.history.replaceState(null,'','/?architecture=distributed')
  vi.stubGlobal('fetch',vi.fn(async()=>({ok:true,json:async()=>({events:[]})})))
  latest.session=session({architecture:'distributed',status:'delivery_unknown',handoffs:[{...handoff,architecture:'distributed',session_id:'test-session',actual_execution_status:'delivery_unknown',receiver_receipt:null}]})
  await ready()
  expect(screen.queryByRole('button',{name:'B 已接收 KV V1'})).not.toBeInTheDocument()
  expect(screen.getAllByText(/提交结果未知/).length).toBeGreaterThan(0)
})

it('treats old Distributed URLs as read only and offers return to Local', async () => {
  window.history.replaceState(null, '', '/?architecture=distributed')
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => ({ events: [] }) })))
  render(<DiscussionPanel />)
  await waitFor(() => expect(getLatestDiscussion).toHaveBeenCalledWith('distributed'))
  expect(screen.queryByLabelText('运行架构')).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '开始讨论' })).toBeDisabled()
  fireEvent.click(screen.getByRole('button', { name: '返回本地实验' }))
  await waitFor(() => expect(getLatestDiscussion).toHaveBeenCalledWith('local'))
  expect(window.location.search).not.toContain('architecture=distributed')
})

it('offers four rounds by default, ten normally and twenty with explicit advanced settings', async () => {
  await ready()
  configure()
  expect(screen.getByLabelText('讨论轮数')).toHaveValue('4')
  expect(within(screen.getByLabelText('讨论轮数')).getAllByRole('option')).toHaveLength(10)
  fireEvent.click(screen.getByLabelText('高级轮数（最多20轮；先检查上下文与资源）'))
  change('讨论轮数','20')
  change('每次发言最大 Tokens','1')
  change('总上下文上限 Tokens','32768')
  start()
  await waitFor(() => expect(startDiscussion).toHaveBeenCalledWith(expect.objectContaining({rounds:20,advanced_rounds:true,max_context_tokens:32768,max_new_tokens:1}),expect.any(String)))
})

it('renders a compact idealized subtitle without block risk styling', async () => {
  latest.session=session({config:{...session().config,rounds:3,mode:'auto',idealized_simulation:true}})
  await ready()
  expect(screen.getByText('3轮交流 · AUTO')).toBeInTheDocument()
  const context=document.querySelector('.experiment-heading .context')!
  expect(context.textContent).toBe('3轮交流 · AUTO · 理想化仿真')
  expect(context.querySelector('.risk-note')).toBeNull()
})

it('running idealized sessions expose frozen assumptions without editable coefficients', async () => {
  latest.session=session({status:'running',config:{...session().config,mode:'auto',idealized_simulation:true,synthetic_assumptions:{schema:'synthetic_compute_v1',source:'synthetic_assumption',units:'ms',a:31,b:2,c:.004,d:5,e:.05,f:.2}}})
  await ready(); configure()
  expect(screen.getByText(/TOKEN = 31 \+ 2N \+ 0.004NH/)).toBeInTheDocument()
  expect(screen.queryByLabelText('假设系数 a')).not.toBeInTheDocument()
  expect(screen.queryByRole('button',{name:'恢复默认假设'})).not.toBeInTheDocument()
})

it('labels measured profiles by actual provenance in the subtitle', async () => {
  latest.session=session({config:{...session().config,mode:'auto',cost_profile:{...inlineProfile,provenance:'measured_incremental'}}})
  await ready()
  expect(document.querySelector('.experiment-heading .context')).toHaveTextContent('实测 Profile')
  expect(document.querySelector('.experiment-heading .context')).not.toHaveTextContent('Synthetic')
})


it('keeps edited AUTO drafts across tabs and close, then submits all custom coefficients', async () => {
  await ready(); configure('通信')
  expect(screen.getByLabelText('通信模式')).toHaveValue('auto')
  expect(screen.getByLabelText('AUTO 的 KV 版本')).toHaveValue('v0')
  expect(screen.queryByLabelText('AUTO 策略')).not.toBeInTheDocument()
  expect(screen.getByLabelText('理想化场景仿真')).toBeChecked()
  fireEvent.click(screen.getByText('高级：计算成本假设'))
  for (const k of ['a','b','c','d','e','f']) change(`假设系数 ${k}`, '7')
  change('AUTO 的 KV 版本', 'v2')
  configure('网络'); configure('任务'); configure('通信')
  change('通信模式', 'token'); change('通信模式', 'auto')
  fireEvent.click(dialog().getByRole('button', { name: '保留草稿' })); configure('通信')
  expect(screen.getByLabelText('AUTO 的 KV 版本')).toHaveValue('v2')
  expect(document.querySelector('.modal-foot')).not.toHaveTextContent('Synthetic')
  start()
  await waitFor(() => expect(startDiscussion).toHaveBeenCalledWith(expect.objectContaining({mode:'auto', auto_policy:'performance_experiment', auto_kv_version:'v2', idealized_simulation:true, synthetic_assumptions:expect.objectContaining({a:7,b:7,c:7,d:7,e:7,f:7})}),expect.any(String)))
})

it('resets new-experiment communication defaults without changing the historical session', async () => {
  const old = session({config:{...session().config, mode:'auto', auto_policy:'quality_first', idealized_simulation:false, auto_kv_version:'v2'}})
  latest.session = old
  const frozen = JSON.stringify(old)
  await ready(); configure('通信')
  expect(screen.getByLabelText('通信模式')).toHaveValue('auto')
  expect(screen.getByLabelText('AUTO 的 KV 版本')).toHaveValue('v0')
  expect(screen.getByLabelText('理想化场景仿真')).toBeChecked()
  change('通信模式','token'); closeConfig()
  fireEvent.click(screen.getByRole('button', {name:'新实验'})); configure('通信')
  expect(screen.getByLabelText('通信模式')).toHaveValue('auto')
  expect(screen.getByLabelText('AUTO 的 KV 版本')).toHaveValue('v0')
  expect(screen.getByLabelText('理想化场景仿真')).toBeChecked()
  expect(screen.queryByLabelText('AUTO 策略')).not.toBeInTheDocument()
  expect(JSON.stringify(old)).toBe(frozen)
  await waitFor(() => expect(dialog().getByRole('button',{name:'开始讨论'})).toBeEnabled())
  start()
  await waitFor(() => expect(startDiscussion).toHaveBeenCalledWith(expect.objectContaining({auto_policy:'performance_experiment', auto_kv_version:'v0',idealized_simulation:true}),expect.any(String)))
})
