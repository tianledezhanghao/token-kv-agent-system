import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from 'react'

import {
  checkHealth,
  getDashboardState,
  getRecentEvents,
  triggerCommunication,
  updateNetwork,
} from '../api/dashboard'
import { useEventStream } from '../hooks/useEventStream'
import {
  dashboardReducer,
  initialDashboardState,
} from '../state/dashboardReducer'
import type {
  AgentState,
  CommunicationMode,
  DashboardEvent,
  ModeSelection,
  AdaptiveExecution,
} from '../types/dashboard'

const fallbackAgents: AgentState[] = [
  { agent_id: 'Agent A', status: 'unknown' },
  { agent_id: 'Agent B', status: 'unknown' },
]

function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value))
    return '—'
  return value.toLocaleString(undefined, { maximumFractionDigits: digits })
}

function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes))
    return '—'
  if (bytes < 1_000) return `${formatNumber(bytes, 0)} B`
  if (bytes < 1_000_000) return `${formatNumber(bytes / 1_000, 1)} KB`
  return `${formatNumber(bytes / 1_000_000, 1)} MB`
}

function formatLatency(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? '—'
    : `${formatNumber(value)} ms`
}

const verification = (value: boolean | null | undefined) =>
  value === true ? 'Verified' : value === false ? 'Failed' : '—'

function formatTime(timestamp: string): string {
  const parsed = new Date(timestamp)
  return Number.isNaN(parsed.getTime())
    ? '—'
    : parsed.toLocaleTimeString([], { hour12: false })
}

function displayMode(mode: CommunicationMode | null | undefined): string {
  if (mode === 'KV') return 'KV CACHE'
  return mode ?? '尚未选择'
}

const agentStatusLabels: Record<string, string> = {
  idle: '空闲',
  generating: '生成中',
  sending: '发送中',
  waiting: '等待中',
  receiving: '接收中',
  unknown: '未知',
}

const eventTypeLabels: Record<string, string> = {
  NETWORK_STATE_CHANGED: '网络状态已变更',
  AGENT_GENERATION_STARTED: 'Agent 开始生成',
  AGENT_GENERATION_FINISHED: 'Agent 生成完成',
  MODE_SELECTED: '模式选择完成',
  TRANSMISSION_STARTED: '传输开始',
  TRANSMISSION_FINISHED: '传输完成',
  REAL_TOKEN_HANDOFF_STARTED: '真实 Token handoff 开始',
  TOKEN_SERIALIZED: 'Token 序列化完成',
  TOKEN_TRANSFERRED: 'Token bytes 传递完成',
  TOKEN_DESERIALIZED: 'Token 反序列化完成',
  RECEIVER_PREFILL_FINISHED: 'Receiver Prefill 完成',
  REAL_TOKEN_HANDOFF_FINISHED: '真实 Token handoff 完成',
  REAL_TOKEN_HANDOFF_FAILED: '真实 Token handoff 失败',
  REAL_KV_HANDOFF_STARTED: '真实 KV handoff 开始',
  KV_PREPARED: 'Sender KV 准备完成',
  KV_SERIALIZED: 'KV 序列化完成',
  KV_TRANSFERRED: 'KV bytes 传递完成',
  KV_DESERIALIZED: 'KV 反序列化完成',
  KV_RESTORED: 'Receiver KV 恢复完成',
  REAL_KV_HANDOFF_FINISHED: '真实 KV handoff 完成',
  REAL_KV_HANDOFF_FAILED: '真实 KV handoff 失败',
  REAL_ADAPTIVE_HANDOFF_STARTED: '真实自适应 handoff 开始',
  REAL_SELECTION_MADE: '真实选择完成',
  SELECTED_PATH_STARTED: '选中路径开始执行',
  REAL_ADAPTIVE_HANDOFF_FINISHED: '真实自适应 handoff 完成',
  REAL_ADAPTIVE_HANDOFF_FAILED: '真实自适应 handoff 失败',
}

function controlErrorMessage(error: unknown, fallback: string): string {
  if (!(error instanceof Error)) return fallback
  const knownMessages: Record<string, string> = {
    'communication is already running': '通信正在运行，请等待本轮结束。',
    'interactive mode is required for control operations':
      '控制操作需要使用交互实验模式启动后端。',
    'network update failed': '网络更新失败。',
  }
  return knownMessages[error.message] ?? error.message
}

function ModeBadge({ mode }: { mode: CommunicationMode | null | undefined }) {
  return (
    <span className={`mode-badge mode-${mode?.toLowerCase() ?? 'none'}`}>
      {displayMode(mode)}
    </span>
  )
}

function eventSummary(event: DashboardEvent): string {
  switch (event.type) {
    case 'NETWORK_STATE_CHANGED':
      return `${formatNumber(event.metadata.bandwidth_mbps as number)} Mbps · 基础时延 ${formatNumber(event.metadata.base_latency_ms as number)} ms`
    case 'MODE_SELECTED':
      return `${displayMode(event.mode)} · TOKEN ${formatNumber(event.metadata.estimated_token_latency_ms as number)} ms · KV ${formatNumber(event.metadata.estimated_kv_latency_ms as number)} ms`
    case 'TRANSMISSION_STARTED':
    case 'TRANSMISSION_FINISHED':
      return `${displayMode(event.mode)} · ${formatBytes(event.payload_bytes)} · ${formatLatency(event.transmission_latency_ms)}`
    case 'AGENT_GENERATION_STARTED':
      return typeof event.metadata.prompt === 'string'
        ? event.metadata.prompt
        : '开始生成'
    case 'AGENT_GENERATION_FINISHED':
      return event.metadata.real_token === true ||
        event.metadata.real_kv === true ||
        event.metadata.real_adaptive === true
        ? `${formatNumber(event.metadata.token_count as number, 0)} tokens · 真实模型输出`
        : `${formatNumber(event.metadata.token_count as number, 0)} tokens · KV 估算 ${formatBytes(event.metadata.kv_payload_bytes as number)}`
    case 'TOKEN_SERIALIZED':
      return `${formatNumber(event.metadata.token_count as number, 0)} tokens · ${formatBytes(event.metadata.wire_bytes as number)} · ${formatLatency(event.metadata.serialization_ms as number)}`
    case 'TOKEN_TRANSFERRED':
      return `${formatBytes(event.metadata.wire_bytes as number)} · InMemoryByteTransport`
    case 'TOKEN_DESERIALIZED':
      return `${formatNumber(event.metadata.token_count as number, 0)} tokens · ${formatLatency(event.metadata.deserialization_ms as number)}`
    case 'RECEIVER_PREFILL_FINISHED':
      return `${formatLatency(event.metadata.receiver_prefill_ms as number)} · Identity ${event.metadata.token_identity_verified === true ? 'verified' : 'failed'}`
    case 'REAL_TOKEN_HANDOFF_STARTED':
      return 'TOKEN · direct causal continuation · single-run measured'
    case 'REAL_TOKEN_HANDOFF_FINISHED':
      return `${formatNumber(event.metadata.token_count as number, 0)} tokens · ${formatBytes(event.metadata.wire_bytes as number)} · total ${formatLatency(event.metadata.token_handoff_cost_ms as number)}`
    case 'REAL_TOKEN_HANDOFF_FAILED':
      return String(event.metadata.error ?? '未知错误')
    case 'REAL_KV_HANDOFF_STARTED':
      return 'KV · canonical message span · single-run measured'
    case 'KV_PREPARED':
      return `${formatNumber(event.metadata.token_count as number, 0)} tokens · ${formatLatency(event.metadata.sender_kv_prepare_ms as number)}`
    case 'KV_SERIALIZED':
      return `${formatBytes(event.metadata.wire_total_bytes as number)} · ${formatLatency(event.metadata.total_serialize_ms as number)}`
    case 'KV_TRANSFERRED':
      return `${formatBytes(event.metadata.wire_total_bytes as number)} · InMemoryByteTransport`
    case 'KV_DESERIALIZED':
      return formatLatency(event.metadata.deserialize_ms as number)
    case 'KV_RESTORED':
      return `${formatLatency(event.metadata.total_restore_ms as number)} · Prefix Prefill skipped`
    case 'REAL_KV_HANDOFF_FINISHED':
      return `${formatNumber(event.metadata.token_count as number, 0)} tokens · ${formatBytes(event.metadata.wire_bytes as number)} · E2E ${formatLatency(event.metadata.message_ready_e2e_cost_ms as number)}`
    case 'REAL_KV_HANDOFF_FAILED':
      return String(event.metadata.error ?? '未知错误')
    case 'REAL_SELECTION_MADE':
      return `TOKEN Estimated ${formatLatency((event.metadata.token_estimate as { total_ms?: number } | undefined)?.total_ms)} · KV Estimated ${formatLatency((event.metadata.kv_estimate as { message_ready_total_ms?: number } | undefined)?.message_ready_total_ms)} · Selected ${event.metadata.selected_mode}`
    case 'SELECTED_PATH_STARTED':
      return `Selected ${event.metadata.selected_mode} Data Plane`
    case 'REAL_ADAPTIVE_HANDOFF_FINISHED':
      return `${event.metadata.actually_executed_mode} actual ${formatLatency(event.metadata.actual_selected_cost_ms as number)} · Receiver continued`
    case 'REAL_ADAPTIVE_HANDOFF_FAILED':
      return String(event.metadata.error ?? '未知错误')
    default:
      return '已收到事件'
  }
}

function AdaptiveDecision({
  selection,
  execution,
}: {
  selection: ModeSelection | null | undefined
  execution: AdaptiveExecution | null | undefined
}) {
  const token = selection?.token_estimate
  const kv = selection?.kv_estimate
  const measurement = execution?.selected_measurement
  const actualPathLabel = (mode: 'TOKEN' | 'KV') =>
    execution?.executed_mode === mode
      ? formatLatency(execution.actual_ms)
      : (selection?.selected_mode ?? selection?.mode) === mode
        ? '尚无已确认结果'
        : selection?.mode
          ? 'Not executed'
          : '尚未选择'
  const actual = (key: string) =>
    typeof measurement?.[key] === 'number'
      ? formatLatency(measurement[key] as number)
      : '—'
  return (
    <div className="adaptive-decision">
      <p>
        <strong>Selector: Active · Real Cost Based</strong>
      </p>
      <dl className="cost-breakdown">
        <div>
          <dt>Canonical Tokens</dt>
          <dd>{formatNumber(selection?.token_count, 0)}</dd>
        </div>
        <div>
          <dt>Network Bandwidth</dt>
          <dd>
            {selection ? `${formatNumber(selection.bandwidth_mbps)} Mbps` : '—'}
          </dd>
        </div>
        <div>
          <dt>Base Latency</dt>
          <dd>{formatLatency(selection?.base_latency_ms)}</dd>
        </div>
        <div>
          <dt>Cost Boundary</dt>
          <dd>{selection?.cost_boundary ?? 'MESSAGE_READY'}</dd>
        </div>
      </dl>
      <div className="adaptive-cost-grid">
        <div>
          <h3>TOKEN Estimated</h3>
          <dl className="cost-breakdown">
            <div>
              <dt>Wire Bytes</dt>
              <dd>{formatBytes(token?.wire_bytes)}</dd>
            </div>
            <div>
              <dt>Serialize Estimate</dt>
              <dd>{formatLatency(token?.serialize_ms)}</dd>
            </div>
            <div>
              <dt>Link Estimate</dt>
              <dd>{formatLatency(token?.link_ms)}</dd>
            </div>
            <div>
              <dt>Deserialize Estimate</dt>
              <dd>{formatLatency(token?.deserialize_ms)}</dd>
            </div>
            <div>
              <dt>Receiver Prefill Estimate</dt>
              <dd>{formatLatency(token?.receiver_prefill_ms)}</dd>
            </div>
            <div>
              <dt>TOKEN Estimated Total</dt>
              <dd>{formatLatency(token?.total_ms)}</dd>
            </div>
          </dl>
          <p>TOKEN Actual: {actualPathLabel('TOKEN')}</p>
        </div>
        <div>
          <h3>KV Estimated</h3>
          <dl className="cost-breakdown">
            <div>
              <dt>Raw KV Estimate</dt>
              <dd>{formatBytes(kv?.raw_kv_bytes)}</dd>
            </div>
            <div>
              <dt>Wire KV Estimate</dt>
              <dd>{formatBytes(kv?.wire_bytes)}</dd>
            </div>
            <div>
              <dt>Metadata Estimate</dt>
              <dd>{formatBytes(kv?.wire_metadata_bytes)}</dd>
            </div>
            <div>
              <dt>Sender Prepare Estimate</dt>
              <dd>{formatLatency(kv?.sender_prepare_ms)}</dd>
            </div>
            <div>
              <dt>Serialize Estimate</dt>
              <dd>{formatLatency(kv?.serialize_ms)}</dd>
            </div>
            <div>
              <dt>Link Estimate</dt>
              <dd>{formatLatency(kv?.link_ms)}</dd>
            </div>
            <div>
              <dt>Deserialize Estimate</dt>
              <dd>{formatLatency(kv?.deserialize_ms)}</dd>
            </div>
            <div>
              <dt>Restore Estimate</dt>
              <dd>{formatLatency(kv?.restore_ms)}</dd>
            </div>
            <div>
              <dt>KV Estimated Total</dt>
              <dd>{formatLatency(kv?.message_ready_total_ms)}</dd>
            </div>
          </dl>
          <p>KV Actual: {actualPathLabel('KV')}</p>
        </div>
      </div>
      <div className="selected-mode">
        <span>Decision</span>
        <strong>
          {selection?.selected_mode
            ? `Selected ${selection.selected_mode}`
            : '尚未选择'}
        </strong>
      </div>
      <p>
        Selected Estimated Saving:{' '}
        {formatLatency(selection?.estimated_saving_ms)}
      </p>
      <h3>Actual Execution</h3>
      {execution ? (
        <dl className="cost-breakdown">
          <div>
            <dt>Executed Mode</dt>
            <dd>{execution.executed_mode}</dd>
          </div>
          <div>
            <dt>Predicted</dt>
            <dd>{formatLatency(execution.predicted_ms)}</dd>
          </div>
          <div>
            <dt>Actual MESSAGE_READY Total</dt>
            <dd>{formatLatency(execution.actual_ms)}</dd>
          </div>
          <div>
            <dt>Absolute Error</dt>
            <dd>{formatLatency(execution.absolute_error_ms)}</dd>
          </div>
          <div>
            <dt>Relative Error</dt>
            <dd>{formatNumber(execution.relative_error * 100)}%</dd>
          </div>
          <div>
            <dt>Wire Bytes</dt>
            <dd>
              {formatBytes(measurement?.wire_bytes as number | undefined)}
            </dd>
          </div>
          {execution.executed_mode === 'KV' && (
            <>
              <div>
                <dt>Raw KV</dt>
                <dd>{formatBytes(measurement?.raw_kv_bytes as number)}</dd>
              </div>
              <div>
                <dt>Wire KV</dt>
                <dd>{formatBytes(measurement?.wire_bytes as number)}</dd>
              </div>
              <div>
                <dt>Metadata Overhead</dt>
                <dd>
                  {formatBytes(measurement?.wire_metadata_bytes as number)}
                </dd>
              </div>
              <div>
                <dt>Device → Host</dt>
                <dd>{actual('device_to_host_ms')}</dd>
              </div>
              <div>
                <dt>Encode</dt>
                <dd>{actual('encode_ms')}</dd>
              </div>
              <div>
                <dt>CPU Reconstruction</dt>
                <dd>{actual('cpu_reconstruction_ms')}</dd>
              </div>
              <div>
                <dt>Host → Device</dt>
                <dd>{actual('host_to_device_ms')}</dd>
              </div>
              <div>
                <dt>Cache Construction</dt>
                <dd>{actual('cache_construction_ms')}</dd>
              </div>
              <div>
                <dt>Receiver Prefix Prefill</dt>
                <dd>
                  {measurement?.receiver_prefill_skipped
                    ? 'Skipped'
                    : 'Unexpectedly executed'}
                </dd>
              </div>
            </>
          )}
          {execution.executed_mode === 'KV' && (
            <div>
              <dt>Sender KV Prepare</dt>
              <dd>{actual('sender_prepare_ms')}</dd>
            </div>
          )}
          <div>
            <dt>Serialization</dt>
            <dd>{actual('serialize_ms')}</dd>
          </div>
          <div>
            <dt>Simulated Link</dt>
            <dd>{actual('link_ms')}</dd>
          </div>
          <div>
            <dt>Deserialization</dt>
            <dd>{actual('deserialize_ms')}</dd>
          </div>
          {execution.executed_mode === 'TOKEN' ? (
            <div>
              <dt>Receiver Prefill</dt>
              <dd>{actual('receiver_prefill_ms')}</dd>
            </div>
          ) : (
            <div>
              <dt>Restore</dt>
              <dd>{actual('restore_ms')}</dd>
            </div>
          )}
          <div>
            <dt>Receiver Continuation</dt>
            <dd>{verification(execution.correctness)}</dd>
          </div>
        </dl>
      ) : (
        <p>等待选中路径完成。</p>
      )}
    </div>
  )
}

import {
  CommunicationStage,
  Connection,
  EventLog,
  Icon,
  Modal,
  ReadingSurface,
  Tabs,
} from '../components/Workspace'
export default function CanonicalDashboard() {
  const [modal, setModal] = useState<'config' | 'details' | null>(null)
  const [detailTab, setDetailTab] = useState('costs')
  const controlBusy = useRef(false)
  const loadVersion = useRef(0)
  const duringLoad = useRef<DashboardEvent[] | null>(null)
  const [dashboard, dispatch] = useReducer(
    dashboardReducer,
    initialDashboardState,
  )
  const [loading, setLoading] = useState(true)
  const [backendAvailable, setBackendAvailable] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [controlError, setControlError] = useState<string | null>(null)
  const [controlMessage, setControlMessage] = useState<string | null>(null)
  const [bandwidthDraft, setBandwidthDraft] = useState('20')
  const [networkUpdating, setNetworkUpdating] = useState(false)
  const [communicationRunning, setCommunicationRunning] = useState(false)

  const loadInitialData = useCallback(async (showLoading = false) => {
    const version = ++loadVersion.current
    duringLoad.current = []
    if (showLoading) setLoading(true)
    try {
      await checkHealth()
      const [snapshot, eventResponse] = await Promise.all([
        getDashboardState(),
        getRecentEvents(100),
      ])
      if (version !== loadVersion.current) return
      dispatch({
        type: 'hydrate',
        snapshot,
        events: eventResponse.events,
        pendingEvents: duringLoad.current ?? [],
      })
      duringLoad.current = null
      setBackendAvailable(true)
      setError(null)
    } catch {
      if (version !== loadVersion.current) return
      duringLoad.current = null
      setBackendAvailable(false)
      setError('后端当前不可用。请启动 Dashboard Backend；页面会继续自动重连。')
    } finally {
      if (version === loadVersion.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadInitialData(true)
  }, [loadInitialData])

  const receiveEvent = useCallback((event: DashboardEvent) => {
    if (duringLoad.current) {
      duringLoad.current.push(event)
      if (duringLoad.current.length > 100) duringLoad.current.shift()
    }
    dispatch({ type: 'event', event })
    setBackendAvailable(true)
    setError(null)
  }, [])

  const connectionStatus = useEventStream({
    enabled: !loading,
    onEvent: receiveEvent,
    onReconnect: () => void loadInitialData(),
  })

  const snapshot = dashboard.snapshot
  const runtime = snapshot?.runtime
  const isRealToken = runtime?.mode === 'real_token'
  const isRealKV = runtime?.mode === 'real_kv'
  const isRealAdaptive = runtime?.mode === 'real_adaptive'
  const isRealRuntime = isRealToken || isRealKV || isRealAdaptive
  const isInteractive = runtime?.mode === 'interactive' || isRealRuntime
  const isRunning =
    communicationRunning ||
    runtime?.running === true ||
    snapshot?.system_status === 'running' ||
    Object.values(snapshot?.agents ?? {}).some((a) =>
      ['generating', 'sending', 'receiving', 'waiting'].includes(a.status),
    )
  const agents = snapshot ? Object.values(snapshot.agents) : fallbackAgents
  const displayedAgents =
    agents.length >= 2 ? agents.slice(0, 2) : fallbackAgents
  const selection = snapshot?.last_selection
  const communication = snapshot?.last_communication
  const realMetrics = snapshot?.real_token_metrics
  const realKVMetrics = snapshot?.real_kv_metrics
  const adaptiveExecution = snapshot?.adaptive_execution
  const source = isRunning
    ? (displayedAgents.find((a) => ['generating', 'sending'].includes(a.status))
        ?.agent_id ?? runtime?.next_source)
    : (communication?.source ?? selection?.source)
  const target = isRunning
    ? displayedAgents.find((a) => a.agent_id !== source)?.agent_id
    : (communication?.target ?? selection?.target)
  const mode = isRunning
    ? (runtime?.active_communication_mode ?? selection?.mode)
    : (runtime?.active_communication_mode ??
      selection?.mode ??
      communication?.mode)

  useEffect(() => {
    if (snapshot?.network) {
      setBandwidthDraft(String(snapshot.network.bandwidth_mbps))
    }
  }, [snapshot?.network?.bandwidth_mbps])

  const parsedBandwidth = Number(bandwidthDraft)
  const bandwidthIsValid =
    Number.isFinite(parsedBandwidth) && parsedBandwidth > 0
  const bandwidthChanged =
    isInteractive &&
    !!snapshot?.network &&
    bandwidthIsValid &&
    parsedBandwidth !== snapshot?.network?.bandwidth_mbps

  const applyBandwidth = async () => {
    if (controlBusy.current || isRunning || networkUpdating) return
    if (!bandwidthIsValid) {
      setControlError('带宽必须是大于 0 Mbps 的数字。')
      return
    }
    controlBusy.current = true
    setNetworkUpdating(true)
    setControlError(null)
    setControlMessage(null)
    try {
      await updateNetwork(parsedBandwidth)
      await loadInitialData()
      setControlMessage(
        `网络带宽已更新为 ${formatNumber(parsedBandwidth)} Mbps。`,
      )
    } catch (requestError) {
      setControlError(controlErrorMessage(requestError, '网络更新失败。'))
    } finally {
      controlBusy.current = false
      setNetworkUpdating(false)
    }
  }

  const runCommunication = async () => {
    if (
      controlBusy.current ||
      isRunning ||
      networkUpdating ||
      !backendAvailable
    )
      return
    controlBusy.current = true
    setCommunicationRunning(true)
    setControlError(null)
    setControlMessage(null)
    try {
      const result = await triggerCommunication()
      await loadInitialData()
      setControlMessage(
        isRealToken
          ? `${result.source} → ${result.target} 真实 Token handoff 完成。`
          : isRealKV
            ? `${result.source} → ${result.target} canonical KV handoff 完成。`
            : `${result.source} → ${result.target} 通信完成；Selector 自动选择 ${displayMode(result.selected_mode)}。`,
      )
    } catch (requestError) {
      setControlError(controlErrorMessage(requestError, '通信触发失败。'))
    } finally {
      controlBusy.current = false
      setCommunicationRunning(false)
    }
  }

  const conversations = useMemo(
    () =>
      dashboard.events
        .filter(
          (event) =>
            event.type === 'AGENT_GENERATION_FINISHED' &&
            typeof event.metadata.text === 'string',
        )
        .slice(-4)
        .reverse(),
    [dashboard.events],
  )

  return (
    <>
      <Connection
        status={connectionStatus}
        retry={error ? () => void loadInitialData() : undefined}
      />
      <main className="workspace" aria-label="通信基准工作区">
        <section className="experiment-heading">
          <div className="title-block">
            <h1>通信基准</h1>
            <p className="context">
              {isRealAdaptive
                ? 'Real Adaptive'
                : isRealToken
                  ? 'Real Token'
                  : isRealKV
                    ? 'Real KV'
                    : runtime?.mode === 'observer'
                      ? '只读观察'
                      : 'Mock 模拟'}{' '}
              ·{' '}
              {runtime?.mode === 'auto'
                ? '自动演示'
                : isInteractive
                  ? '交互实验'
                  : '只读观察'}
              {isRealKV ? ' · Canonical KV' : ''}
            </p>
          </div>
          <div className="actions">
            <button className="button" onClick={() => setModal('config')}>
              <Icon name="settings" />
              配置
            </button>
            {isInteractive && (
              <button
                className="button primary"
                disabled={isRunning || networkUpdating || !backendAvailable}
                onClick={() => void runCommunication()}
              >
                {isRunning ? '通信运行中…' : '运行一次通信'}
              </button>
            )}
          </div>
        </section>
        <ReadingSurface
          stage={
            <CommunicationStage
              agents={displayedAgents.map((a) => ({
                name: a.agent_id,
                state: agentStatusLabels[a.status] ?? a.status,
              }))}
              status={
                loading
                  ? '正在加载状态'
                  : isRunning
                    ? '通信进行中'
                    : snapshot?.system_status === 'error'
                      ? '通信失败'
                      : communication
                        ? '最近通信已完成'
                        : '等待通信'
              }
              source={source}
              target={target}
              pending={isRunning}
              mode={
                isRunning
                  ? `本次请求 ${displayMode(mode)} · 待完成`
                  : `最近交接 ${displayMode(communication?.mode)}`
              }
              onEvidence={
                communication
                  ? () => {
                      setDetailTab('recent')
                      setModal('details')
                    }
                  : undefined
              }
            />
          }
          tabs={[{ value: 'conversation', label: 'Agent 内容' }]}
          selected="conversation"
          onSelect={() => {}}
          onDetails={() => {
            setDetailTab('costs')
            setModal('details')
          }}
          viewKey="canonical"
          contentKey={`${conversations.length}:${realMetrics?.sender_generated_text}:${realKVMetrics?.sender_generated_text}:${adaptiveExecution?.sender_text}`}
          footer={
            isRealRuntime
              ? '真实模型 · 实测、模拟与派生成本在详情中分别标注'
              : 'Mock 数据 · 不代表真实模型实验'
          }
        >
          {error && (
            <p className="error-banner" role="alert">
              {error}
            </p>
          )}
          {controlError && modal !== 'config' && (
            <p className="error-banner" role="alert">
              {controlError}
            </p>
          )}
          {controlMessage && modal !== 'config' && (
            <p className="control-feedback" role="status">
              {controlMessage}
            </p>
          )}
          {bandwidthChanged && (
            <p className="hint">
              有未应用的带宽草稿。本次实验仍使用服务器的{' '}
              {snapshot?.network?.bandwidth_mbps} Mbps。
            </p>
          )}
          <section
            className="canonical-conversation"
            aria-labelledby="conversation-title"
          >
            <div className="section-heading">
              <p className="eyebrow">后端事件内容</p>
              <h2 id="conversation-title">Agent 对话</h2>
            </div>
            {isRealAdaptive && adaptiveExecution ? (
              <div className="conversation-list">
                <article>
                  <div>
                    <strong>{adaptiveExecution.source} output</strong>
                  </div>
                  <p>{adaptiveExecution.sender_text}</p>
                </article>
                <article>
                  <div>
                    <strong>{adaptiveExecution.target} continuation</strong>
                  </div>
                  <p>{adaptiveExecution.receiver_text}</p>
                </article>
              </div>
            ) : isRealKV && realKVMetrics ? (
              <div className="conversation-list">
                <article>
                  <div>
                    <strong>{realKVMetrics.source} output</strong>
                  </div>
                  <p>{realKVMetrics.sender_generated_text}</p>
                </article>
                <article>
                  <div>
                    <strong>{realKVMetrics.target} continuation</strong>
                  </div>
                  <p>{realKVMetrics.receiver_generated_text}</p>
                </article>
              </div>
            ) : isRealToken && realMetrics ? (
              <div className="conversation-list">
                <article>
                  <div>
                    <strong>{realMetrics.source} output</strong>
                  </div>
                  <p>{realMetrics.sender_generated_text}</p>
                </article>
                <article>
                  <div>
                    <strong>{realMetrics.target} output</strong>
                  </div>
                  <p>{realMetrics.receiver_generated_text}</p>
                </article>
              </div>
            ) : conversations.length ? (
              <div className="conversation-list">
                {conversations.map((event, index) => (
                  <article key={`${event.timestamp}-${index}`}>
                    <div>
                      <strong>{event.source}</strong>
                      <time>{formatTime(event.timestamp)}</time>
                    </div>
                    <p>{String(event.metadata.text)}</p>
                  </article>
                ))}
              </div>
            ) : (
              <p className="empty-state">尚无 Agent 生成内容。</p>
            )}
          </section>
        </ReadingSurface>
      </main>
      {modal === 'config' && (
        <Modal
          returnLabel="返回基准"
          title="基准配置"
          onClose={() => setModal(null)}
          description="模拟链路只影响通信基准；双 Agent 讨论配置在独立工作区。"
        >
          <section
            className="detail-section network-panel"
            aria-labelledby="network-title"
          >
            <div className="section-heading">
              <p className="eyebrow">链路条件</p>
              <h2 id="network-title">网络环境</h2>
            </div>
            <dl className="readings">
              <div>
                <dt>带宽</dt>
                <dd>
                  {snapshot?.network
                    ? formatNumber(snapshot.network.bandwidth_mbps)
                    : '等待状态'}
                  <small>{snapshot?.network ? ' Mbps' : ''}</small>
                </dd>
              </div>
              <div>
                <dt>基础时延</dt>
                <dd>
                  {snapshot?.network
                    ? formatNumber(snapshot.network.base_latency_ms)
                    : '—'}
                  <small>{snapshot?.network ? ' ms' : ''}</small>
                </dd>
              </div>
            </dl>
            {isInteractive ? (
              <div className="experiment-controls">
                <div className="experiment-heading">
                  <div>
                    <p className="eyebrow">交互实验</p>
                    <h3>
                      {isRealToken
                        ? '改变真实 Token packet 的模拟链路条件'
                        : isRealKV
                          ? '改变真实 KV packet 的模拟链路条件'
                          : '改变网络条件，由 Selector 自动决策'}
                    </h3>
                  </div>
                  <span className="runtime-pill">
                    {isRealAdaptive
                      ? 'Real Adaptive'
                      : isRealToken
                        ? 'Real Token'
                        : isRealKV
                          ? 'Real KV'
                          : '手动实验模式'}
                  </span>
                </div>
                <label className="bandwidth-control" htmlFor="bandwidth-slider">
                  <span>带宽设置值</span>
                  <output>
                    {bandwidthIsValid ? formatNumber(parsedBandwidth) : '无效'}{' '}
                    Mbps
                  </output>
                </label>
                <input
                  id="bandwidth-slider"
                  aria-label="带宽滑块"
                  type="range"
                  min="1"
                  max="2000"
                  step="1"
                  value={bandwidthIsValid ? Math.min(parsedBandwidth, 2000) : 1}
                  onChange={(event) => setBandwidthDraft(event.target.value)}
                  disabled={isRunning || networkUpdating}
                />
                <div className="bandwidth-entry">
                  <div className="number-with-unit">
                    <input
                      aria-label="带宽 Mbps"
                      type="number"
                      min="0"
                      step="1"
                      value={bandwidthDraft}
                      onChange={(event) =>
                        setBandwidthDraft(event.target.value)
                      }
                      disabled={isRunning || networkUpdating}
                    />
                    <span>Mbps</span>
                  </div>
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => void applyBandwidth()}
                    disabled={!bandwidthChanged || isRunning || networkUpdating}
                  >
                    {networkUpdating ? '正在应用…' : '应用网络设置'}
                  </button>
                </div>
                <div className="run-row">
                  <div>
                    <span>下一次通信</span>
                    <strong>
                      {runtime.next_source} → {runtime.next_target}
                    </strong>
                  </div>
                  <button
                    className="run-button"
                    type="button"
                    onClick={() => void runCommunication()}
                    disabled={isRunning || networkUpdating || !backendAvailable}
                  >
                    {isRunning ? '通信运行中…' : '运行一次通信'}
                  </button>
                </div>
                {isRealRuntime && (
                  <dl className="runtime-details">
                    <div>
                      <dt>Runtime</dt>
                      <dd>
                        {isRealAdaptive
                          ? 'Real Adaptive'
                          : isRealKV
                            ? 'Real KV'
                            : 'Real Token'}
                      </dd>
                    </div>
                    <div>
                      <dt>Model</dt>
                      <dd>{runtime.model ?? '—'}</dd>
                    </div>
                    <div>
                      <dt>Device</dt>
                      <dd>{runtime.device ?? '—'}</dd>
                    </div>
                    {isRealAdaptive && (
                      <>
                        <div>
                          <dt>Selector</dt>
                          <dd>Active · Real Cost Based</dd>
                        </div>
                        <div>
                          <dt>Profile</dt>
                          <dd>
                            {runtime.selector_profile?.profile_name ?? '—'}
                          </dd>
                        </div>
                        <div>
                          <dt>Boundary</dt>
                          <dd>{runtime.cost_boundary ?? '—'}</dd>
                        </div>
                        <div>
                          <dt>Calibration Tokens</dt>
                          <dd>
                            {runtime.selector_profile?.calibration_token_counts.join(
                              ', ',
                            ) ?? '—'}
                          </dd>
                        </div>
                        {runtime.selector_profile?.development_calibration && (
                          <div>
                            <dt>Calibration</dt>
                            <dd>Development calibration</dd>
                          </div>
                        )}
                      </>
                    )}
                  </dl>
                )}
                {controlError && (
                  <p className="control-feedback error" role="alert">
                    {controlError}
                  </p>
                )}
                {controlMessage && (
                  <p className="control-feedback success" role="status">
                    {controlMessage}
                  </p>
                )}
              </div>
            ) : (
              <div className="auto-mode-note">
                <strong>
                  {runtime?.mode === 'auto' ? '自动演示模式' : '只读观察模式'}
                </strong>
                <span>
                  使用 <code>--interactive</code> 启动 Backend
                  后可使用交互控制。
                </span>
              </div>
            )}
          </section>
        </Modal>
      )}
      {modal === 'details' && (
        <Modal
          returnLabel="返回基准"
          title="实验详情"
          drawer
          onClose={() => setModal(null)}
          description={
            isRealRuntime
              ? '真实模型通信 · 保留成本来源与执行边界'
              : 'Mock / 估算数据，不代表真实模型性能'
          }
          tabs={
            <Tabs
              label="基准详情分区"
              value={detailTab}
              onChange={setDetailTab}
              items={[
                { value: 'costs', label: '成本' },
                { value: 'recent', label: '最近通信' },
                { value: 'events', label: '日志' },
                { value: 'diagnostics', label: '诊断' },
              ]}
            />
          }
        >
          {detailTab === 'costs' && (
            <>
              <section
                className="detail-section decision-panel"
                aria-labelledby="decision-title"
              >
                <div className="section-heading">
                  <p className="eyebrow">
                    {isRealRuntime ? '真实通信路径' : 'Selector 输出'}
                  </p>
                  <h2 id="decision-title">
                    {isRealAdaptive
                      ? 'Real Adaptive Decision'
                      : isRealKV
                        ? 'KV 通信成本'
                        : isRealToken
                          ? 'Token 成本明细'
                          : '模式决策'}
                  </h2>
                </div>
                {isRealAdaptive ? (
                  <AdaptiveDecision
                    selection={selection}
                    execution={adaptiveExecution}
                  />
                ) : isRealKV ? (
                  <>
                    <div className="real-mode-note">
                      <div>
                        <span>Communication Mode</span>
                        <ModeBadge mode="KV" />
                      </div>
                      <p>
                        <strong>Selector: Not active</strong>
                        <br />
                        固定 Real KV experiment runtime；使用 canonical
                        message-span semantics。
                      </p>
                    </div>
                    <dl className="cost-breakdown">
                      <div>
                        <dt>
                          Canonical Token Count <small>Canonical span</small>
                        </dt>
                        <dd>{formatNumber(realKVMetrics?.token_count, 0)}</dd>
                      </div>
                      <div>
                        <dt>
                          Raw KV <small>Measured representation</small>
                        </dt>
                        <dd>{formatBytes(realKVMetrics?.raw_kv_bytes)}</dd>
                      </div>
                      <div>
                        <dt>
                          Wire KV <small>Measured wire</small>
                        </dt>
                        <dd>{formatBytes(realKVMetrics?.wire_bytes)}</dd>
                      </div>
                      <div>
                        <dt>
                          Metadata Overhead <small>Measured wire</small>
                        </dt>
                        <dd>
                          {formatBytes(realKVMetrics?.wire_metadata_bytes)}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Sender KV Preparation{' '}
                          <small>Measured · single run</small>
                        </dt>
                        <dd>
                          {formatLatency(realKVMetrics?.sender_kv_prepare_ms)}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Device → Host <small>Measured · single run</small>
                        </dt>
                        <dd>
                          {formatLatency(realKVMetrics?.device_to_host_ms)}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Encode <small>Measured · single run</small>
                        </dt>
                        <dd>
                          {formatLatency(realKVMetrics?.protocol_encode_ms)}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Total Serialize <small>Measured · single run</small>
                        </dt>
                        <dd>
                          {formatLatency(realKVMetrics?.total_serialize_ms)}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Bandwidth <small>Simulated input</small>
                        </dt>
                        <dd>
                          {realKVMetrics
                            ? `${formatNumber(realKVMetrics.bandwidth_mbps)} Mbps`
                            : '—'}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Base Latency <small>Simulated input</small>
                        </dt>
                        <dd>{formatLatency(realKVMetrics?.base_latency_ms)}</dd>
                      </div>
                      <div>
                        <dt>
                          Simulated Link <small>Simulated · no sleep</small>
                        </dt>
                        <dd>
                          {formatLatency(realKVMetrics?.simulated_link_ms)}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Deserialize <small>Measured · single run</small>
                        </dt>
                        <dd>{formatLatency(realKVMetrics?.deserialize_ms)}</dd>
                      </div>
                      <div>
                        <dt>
                          CPU Reconstruction{' '}
                          <small>Measured · single run</small>
                        </dt>
                        <dd>
                          {formatLatency(
                            realKVMetrics?.cpu_tensor_reconstruction_ms,
                          )}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Host → Device <small>Measured · single run</small>
                        </dt>
                        <dd>
                          {formatLatency(realKVMetrics?.host_to_device_ms)}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Cache Construction{' '}
                          <small>Measured · single run</small>
                        </dt>
                        <dd>
                          {formatLatency(realKVMetrics?.cache_construction_ms)}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Total Restore <small>Measured · single run</small>
                        </dt>
                        <dd>
                          {formatLatency(realKVMetrics?.total_restore_ms)}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Representation-ready{' '}
                          <small>Derived · KV ready → restored</small>
                        </dt>
                        <dd>
                          {formatLatency(
                            realKVMetrics?.representation_ready_cost_ms,
                          )}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Message-ready E2E{' '}
                          <small>Derived · includes Sender preparation</small>
                        </dt>
                        <dd>
                          {formatLatency(
                            realKVMetrics?.message_ready_e2e_cost_ms,
                          )}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Receiver Prefix Prefill <small>Correctness</small>
                        </dt>
                        <dd>
                          {realKVMetrics
                            ? realKVMetrics.receiver_prefix_prefill_executed ===
                              true
                              ? 'Unexpectedly executed'
                              : realKVMetrics.receiver_prefix_prefill_executed ===
                                  false
                                ? 'Skipped — restored from KV cache'
                                : '—'
                            : '—'}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Continuation <small>Correctness</small>
                        </dt>
                        <dd>
                          {realKVMetrics
                            ? verification(realKVMetrics.continuation_verified)
                            : '—'}
                        </dd>
                      </div>
                    </dl>
                  </>
                ) : isRealToken ? (
                  <>
                    <div className="real-mode-note">
                      <div>
                        <span>Communication Mode</span>
                        <ModeBadge mode="TOKEN" />
                      </div>
                      <p>
                        <strong>Selector: Not active</strong>
                        <br />
                        固定 Real Token experiment runtime；不运行 adaptive
                        selection。
                      </p>
                    </div>
                    <dl className="cost-breakdown">
                      <div>
                        <dt>
                          Token Count <small>Payload</small>
                        </dt>
                        <dd>{formatNumber(realMetrics?.token_count, 0)}</dd>
                      </div>
                      <div>
                        <dt>
                          Theoretical Token Bytes <small>Payload</small>
                        </dt>
                        <dd>
                          {formatBytes(realMetrics?.theoretical_token_bytes)}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Wire Bytes <small>Measured</small>
                        </dt>
                        <dd>{formatBytes(realMetrics?.wire_bytes)}</dd>
                      </div>
                      <div>
                        <dt>
                          Serialization <small>Measured · single run</small>
                        </dt>
                        <dd>{formatLatency(realMetrics?.serialization_ms)}</dd>
                      </div>
                      <div>
                        <dt>
                          Simulated Link <small>Simulated</small>
                        </dt>
                        <dd>{formatLatency(realMetrics?.simulated_link_ms)}</dd>
                      </div>
                      <div>
                        <dt>
                          Deserialization <small>Measured · single run</small>
                        </dt>
                        <dd>
                          {formatLatency(realMetrics?.deserialization_ms)}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Receiver Prefill <small>Measured · single run</small>
                        </dt>
                        <dd>
                          {formatLatency(realMetrics?.receiver_prefill_ms)}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Token Handoff Cost <small>Derived</small>
                        </dt>
                        <dd>
                          {formatLatency(realMetrics?.token_handoff_cost_ms)}
                        </dd>
                      </div>
                      <div>
                        <dt>
                          Token Identity <small>Correctness</small>
                        </dt>
                        <dd>
                          {realMetrics
                            ? verification(realMetrics.token_identity_verified)
                            : '—'}
                        </dd>
                      </div>
                    </dl>
                  </>
                ) : (
                  <>
                    <div className="decision-options">
                      <div
                        className={
                          mode === 'TOKEN'
                            ? 'decision-row selected'
                            : 'decision-row'
                        }
                      >
                        <span>
                          <b>T</b> TOKEN
                        </span>
                        <strong>
                          {formatLatency(selection?.estimated_token_latency_ms)}
                        </strong>
                      </div>
                      <div
                        className={
                          mode === 'KV'
                            ? 'decision-row selected'
                            : 'decision-row'
                        }
                      >
                        <span>
                          <b>K</b> KV CACHE
                        </span>
                        <strong>
                          {formatLatency(selection?.estimated_kv_latency_ms)}
                        </strong>
                      </div>
                    </div>
                    <div className="selected-mode">
                      <span>最终选择</span>
                      <ModeBadge mode={mode} />
                    </div>
                  </>
                )}
              </section>
              <details>
                <summary>完整成本与选择数据</summary>
                <pre className="raw-log">
                  {JSON.stringify(
                    {
                      selection,
                      realMetrics,
                      realKVMetrics,
                      adaptiveExecution,
                    },
                    null,
                    2,
                  )}
                </pre>
              </details>
            </>
          )}
          {detailTab === 'recent' && (
            <>
              <section
                className="detail-section recent-panel"
                aria-labelledby="recent-title"
              >
                <div className="section-heading">
                  <p className="eyebrow">最近完成的传输</p>
                  <h2 id="recent-title">最近通信</h2>
                </div>
                {communication ? (
                  <div className="communication-details">
                    <div className="route">
                      <strong>{communication.source}</strong>
                      <span>→</span>
                      <strong>{communication.target}</strong>
                      <ModeBadge mode={communication.mode} />
                    </div>
                    <dl>
                      <div>
                        <dt>载荷大小</dt>
                        <dd>{formatBytes(communication.payload_bytes)}</dd>
                      </div>
                      <div>
                        <dt>传输时延</dt>
                        <dd>
                          {formatLatency(communication.transmission_latency_ms)}
                        </dd>
                      </div>
                      <div>
                        <dt>预计端到端</dt>
                        <dd>
                          {formatLatency(
                            communication.estimated_e2e_latency_ms,
                          )}
                        </dd>
                      </div>
                    </dl>
                  </div>
                ) : (
                  <p className="empty-state">尚无通信记录。</p>
                )}
              </section>
              <details>
                <summary>完整通信记录</summary>
                <pre className="raw-log">
                  {JSON.stringify(communication, null, 2)}
                </pre>
              </details>
            </>
          )}
          {detailTab === 'events' && (
            <EventLog
              events={dashboard.events.filter((e) => !e.metadata.session_id)}
              labels={eventTypeLabels}
              summary={eventSummary}
            />
          )}
          {detailTab === 'diagnostics' && (
            <>
              <h3>Runtime 与原始状态</h3>
              <pre className="raw-log">{JSON.stringify(snapshot, null, 2)}</pre>
              <p className="hint">
                数据来自服务端状态快照。Canonical KV 与讨论增量 KV V0/V1
                的语义不同。
              </p>
            </>
          )}
        </Modal>
      )}
    </>
  )
}
