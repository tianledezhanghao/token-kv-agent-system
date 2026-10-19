import { useCallback, useEffect, useRef, useState } from 'react'
import { useEventStream } from '../hooks/useEventStream'
import { getRecentEvents } from '../api/dashboard'
import type { DashboardEvent } from '../types/dashboard'
import {
  getLatestDiscussion,
  switchModel,
  getDiscussion,
  previewDiscussionPrompts,
  startDiscussion,
  patchDiscussionNetwork,
  releaseDiscussion,
} from './api'
import { defaultNetworkAuto, draftFromConfig, networkAutoConfig, parseProfile } from './autoConfig'
import { projectDiscussion, restoreDiscussion } from './state'
import type {
  Architecture,
  NetworkPatch,
  DiscussionConfig,
  DiscussionSession,
  LatestDiscussion,
  ManualMode,
  PromptConfig,
} from './types'

function mergeEvents(first: DashboardEvent[], second: DashboardEvent[]) {
  const byKey = new Map<string, DashboardEvent>()
  for (const e of [...first, ...second])
    byKey.set(
      `${e.metadata.architecture ?? 'local'}:${e.metadata.session_id}:${e.metadata.revision}`,
      e,
    )
  return [...byKey.values()]
    .sort(
      (a, b) =>
        a.metadata.session_id === b.metadata.session_id
          ? Number(a.metadata.revision) - Number(b.metadata.revision)
          : a.timestamp.localeCompare(b.timestamp),
    )
    .slice(-100)
}
export function useDiscussion() {
  const [architecture, setArchitecture] = useState<Architecture>(new URLSearchParams(window.location.search).get('architecture') === 'distributed' ? 'distributed' : 'local')
  const architectureRef = useRef(architecture)
  architectureRef.current = architecture
  const [contextTokens, setContextTokens] = useState('8192')
  const [scenario, setScenario] = useState<'discussion' | 'secret_sum'>(
    'discussion',
  )
  const [secretA, setSecretA] = useState('17')
  const [secretB, setSecretB] = useState('26')
  const secret = scenario === 'secret_sum'
  const [question, setQuestion] = useState('23 + 48 = ?')
  const [rounds, setRounds] = useState(4)
  const [tokens, setTokens] = useState('512')
  const [summarizer, setSummarizer] = useState<'A' | 'B'>('A')
  const [mode, setMode] = useState<DiscussionConfig['mode']>('auto')
  const [plan, setPlan] = useState<ManualMode[]>(Array.from({length:40},(_,i) => i % 2 ? 'kv:v1' : 'token'))
  const [networkAuto, setNetworkAuto] = useState(defaultNetworkAuto)
  const [recomputePolicy, setRecomputePolicy] = useState<'tokens' | 'ratio'>('tokens')
  const [recomputeValue, setRecomputeValue] = useState('1')
  const [agentA, setAgentA] = useState('')
  const [agentB, setAgentB] = useState('')
  const [preview, setPreview] = useState<{
    a: string
    b: string
    config: PromptConfig
  } | null>(null)
  const [promptError, setPromptError] = useState<string | null>(null)
  const [events, setEvents] = useState<DashboardEvent[]>([])
  const [eventsError, setEventsError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [latest, setLatest] = useState<LatestDiscussion | null>(null)
  const [session, setSession] = useState<DiscussionSession | null>(null)
  const sessionRef = useRef(session)
  sessionRef.current = session
  const [notice, setNotice] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [recoveryError, setRecoveryError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const submittingRef = useRef(false)
  const [recovering, setRecovering] = useState(false)
  const selectedSessionId = useRef(new URLSearchParams(window.location.search).get('discussion'))
  const setSessionUrl = (id: string | null) => {
    const url = new URL(window.location.href)
    if (id) url.searchParams.set('discussion', id)
    else url.searchParams.delete('discussion')
    if (architecture === 'distributed') url.searchParams.set('architecture', architecture)
    else url.searchParams.delete('architecture')
    window.history.replaceState(null, '', url)
  }
  const refreshVersion = useRef(0)
  const eventRefreshVersion = useRef(0)
  const draftEdited = useRef(false)

  const refresh = useCallback(async () => {
    const version = ++refreshVersion.current
    const sessionAtRequest = sessionRef.current
    try {
      let data = await getLatestDiscussion(architecture)
      const selectedId = selectedSessionId.current
      if (selectedId) {
        try { data = { ...data, session: await getDiscussion(selectedId, ...(architecture === 'distributed' ? [architecture] as const : [])) } }
        catch (e) {
          if (version !== refreshVersion.current) return
          if (e instanceof Error && e.message.includes('not found')) {
            selectedSessionId.current = null
            setSessionUrl(null)
            setNotice('所选会话已不在服务端最近三次记录内；已恢复最新会话。进程重启也会清空历史。')
          } else throw e
        }
      }
      if (version !== refreshVersion.current) return
      if (
        !data.session &&
        sessionAtRequest &&
        sessionRef.current?.session_id === sessionAtRequest.session_id
      )
        setNotice(
          '服务器当前没有此会话记录，可能已重启。会话仅保存在后端进程内，原草稿仍保留。',
        )
      setLatest(data)
      setLoaded(true)
      setSession((current) =>
        data.session
          ? selectedSessionId.current && current?.session_id !== data.session.session_id ? data.session : restoreDiscussion(current, data.session)
          : current?.session_id === sessionAtRequest?.session_id
            ? null
            : current,
      )
      if (
        data.session &&
        (selectedSessionId.current || restoreDiscussion(sessionRef.current, data.session) === data.session) &&
        !draftEdited.current &&
        !submittingRef.current
      ) {
        const config = data.session.config
        setContextTokens(String(config.max_context_tokens ?? 8192))
        setScenario(config.scenario ?? 'discussion')
        setSecretA(String(config.secret_a ?? 17))
        setSecretB(String(config.secret_b ?? 26))
        setQuestion(config.question)
        setRounds(config.rounds)
        setTokens(String(config.max_new_tokens))
        setSummarizer(config.summarizer)
        // The Session remains frozen; its editable successor starts with new-experiment defaults.
        setMode('auto')
        setNetworkAuto({ ...draftFromConfig(config), autoPolicy: 'performance_experiment', selectedKv: 'v0', idealized: true, synthetic: undefined, allowV3: false })
        setRecomputePolicy(config.kv_recompute_ratio != null ? 'ratio' : 'tokens')
        setRecomputeValue(String(config.kv_recompute_ratio ?? config.kv_recompute_tokens ?? 1))
        setAgentA(config.agent_a_system_prompt ?? '')
        setAgentB(config.agent_b_system_prompt ?? '')
        if (config.mode === 'mixed')
          setPlan((current) =>
            current.map((m, i) => config.handoff_plan[i]?.mode ?? m),
          )
      }
      setRecovering(false)
      setRecoveryError(null)
      return data
    } catch (e) {
      if (version !== refreshVersion.current) return
      setLoaded(true)
      setRecovering(true)
      if (architecture === 'distributed') {
        setLatest({ available: false, model: null, session: sessionRef.current, architecture })
        setRecovering(false)
        setRecoveryError('分布式服务未就绪。启动 run_distributed_dashboard.py 后重试。')
      } else setRecoveryError(e instanceof Error ? e.message : '无法获取服务端讨论状态')
    }
  }, [architecture])
  const refreshEvents = useCallback(async () => {
    const version = ++eventRefreshVersion.current
    try {
      const data = architecture === 'distributed'
        ? await fetch('/api/distributed/events?limit=100').then(r => { if (!r.ok) throw new Error('events unavailable'); return r.json() as Promise<{ events: DashboardEvent[] }> })
        : await getRecentEvents(100)
      if (version !== eventRefreshVersion.current) return
      setEvents((current) => mergeEvents(data.events.filter((e: DashboardEvent) => (e.metadata.architecture ?? 'local') === architecture), current))
      setEventsError(null)
    } catch {
      if (version !== eventRefreshVersion.current) return
      setEventsError('历史日志读取失败；当前仅保留已收到的实时事件。')
    }
  }, [architecture])
  useEffect(() => {
    void refresh()
    void refreshEvents()
  }, [refresh, refreshEvents])
  useEffect(() => {
    let cancelled = false
    setPromptError(null)
    const timer = window.setTimeout(() => {
      void previewDiscussionPrompts(agentA, agentB, architecture)
        .then((config) => {
          if (!cancelled) setPreview({ a: agentA, b: agentB, config })
        })
        .catch((e) => {
          if (!cancelled)
            setPromptError(e instanceof Error ? e.message : 'Prompt 预览失败')
        })
    }, 150)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [agentA, agentB, architecture])
  const onEvent = useCallback(
    (event: DashboardEvent) => {
      if (typeof event.metadata.session_id !== 'string' || (event.metadata.architecture ?? 'local') !== architectureRef.current) return
      setEvents((current) => mergeEvents(current, [event]))
      if (['DISCUSSION_COMPLETED', 'DISCUSSION_FAILED'].includes(event.type))
        setLatest(current => current && current.active_session_id === event.metadata.session_id ? { ...current, active_session_id: null } : current)
      const current = sessionRef.current
      if (current && current.session_id !== event.metadata.session_id && event.type !== 'DISCUSSION_STARTED') return
      if (
        !current ||
        current.session_id !== event.metadata.session_id ||
        Number(event.metadata.revision) > current.revision + 1
      ) {
        setRecovering(true)
        void refresh()
        return
      }
      setSession((s) => projectDiscussion(s, event))
    },
    [refresh],
  )
  const connection = useEventStream({
    enabled: true,
    path: architecture === 'distributed' ? '/ws/distributed/events' : undefined,
    onEvent,
    onReconnect: () => {
      void refresh()
      void refreshEvents()
    },
  })
  useEffect(() => {
    if (architecture !== 'distributed' && connection === 'connected' && !latest?.model_switch && !recovering && (!latest?.active_session_id || latest.active_session_id === session?.session_id)) return
    const timer = window.setInterval(() => void refresh(), 3000)
    return () => window.clearInterval(timer)
  }, [connection, recovering, refresh, latest?.active_session_id, session?.session_id, !!latest?.model_switch, architecture])

  const [switchPending, setSwitchPending] = useState(false)
  const modelLoading = switchPending || !!latest?.model_switch?.loading
  const changeModel = async (size: string) => {
    if (architecture !== 'local') return
    setSwitchPending(true)
    setError(null)
    try { setLatest(await switchModel(size)) }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); await refresh() }
    finally { setSwitchPending(false) }
  }
  const running = modelLoading || submitting || session?.status === 'running' || (!!latest?.active_session_id && latest.active_session_id !== session?.session_id)
  const effectiveRounds = secret ? 1 : rounds
  const previewReady =
    secret || (preview?.a === agentA && preview?.b === agentB && !promptError)
  const hasV2 = (mode === 'auto' && networkAuto.selectedKv === 'v2') || mode === 'kv:v2' || (mode === 'mixed' && plan.slice(0, effectiveRounds * 2).includes('kv:v2'))
  let networkFields: Partial<DiscussionConfig> = {}, configurationError: string | null = null
  try { networkFields = networkAutoConfig(networkAuto, mode, latest?.cost_model_identity) }
  catch (e) { configurationError = e instanceof Error ? e.message : '网络/AUTO 配置无效' }
  let profileProvenance: string | undefined
  try { profileProvenance = mode === 'auto' && !networkAuto.idealized ? parseProfile(networkAuto.profileText)?.provenance : undefined } catch { /* Invalid JSON has inline feedback. */ }
  const valid = architecture === 'local' && rounds >= 1 && rounds <= 20 && !configurationError && (Number.isInteger(Number(contextTokens)) && Number(contextTokens) >= 128 && Number(contextTokens) <= 32768) &&
    (secret
      ? [secretA, secretB].every(
          (n) =>
            n.trim() &&
            Number.isInteger(Number(n)) &&
            Number(n) >= 1 &&
            Number(n) <= 99,
        )
      : question.trim().length > 0 && question.length <= 512) &&
    Number.isInteger(Number(tokens)) &&
    Number(tokens) >= 1 &&
    Number.isSafeInteger(Number(tokens)) &&
    (!hasV2 || (recomputeValue.trim() !== '' && Number.isFinite(Number(recomputeValue)) && Number(recomputeValue) >= 0 &&
      (recomputePolicy === 'tokens' ? Number.isInteger(Number(recomputeValue)) : Number(recomputeValue) <= 1)))
  const submit = async () => {
    if (
      submittingRef.current ||
      running ||
      !valid ||
      !previewReady ||
      !latest?.available ||
      recovering ||
      !loaded
    )
      return
    selectedSessionId.current = null
    setSessionUrl(null)
    submittingRef.current = true
    setSubmitting(true)
    setNotice(null)
    setError(null)
    const config: DiscussionConfig = {
      ...networkFields,
      architecture: 'local', max_context_tokens: Number(contextTokens), advanced_rounds: effectiveRounds > 10,
      question: secret
        ? 'Exchange private numbers and compute their sum.'
        : question,
      rounds: effectiveRounds,
      max_new_tokens: Number(tokens),
      summarizer: secret ? 'A' : summarizer,
      mode,
      ...(hasV2 ? (recomputePolicy === 'tokens' ? { kv_recompute_tokens: Number(recomputeValue) } : { kv_recompute_ratio: Number(recomputeValue) }) : {}),
      ...(secret
        ? {
            scenario: 'secret_sum',
            secret_a: Number(secretA),
            secret_b: Number(secretB),
          }
        : { agent_a_system_prompt: agentA, agent_b_system_prompt: agentB }),
      handoff_plan:
        mode === 'mixed'
          ? plan.slice(0, effectiveRounds * 2).map((m, i) => ({
              round: Math.floor(i / 2) + 1,
              source: i % 2 ? 'B' : 'A',
              target: i % 2 ? 'A' : 'B',
              mode: m,
            }))
          : [],
    }
    const requestedSessionId = crypto.randomUUID()
    let accepted = false
    try {
      const started = await startDiscussion(config, requestedSessionId)
      setSession((s) => restoreDiscussion(s, started))
      accepted = true
    } catch (e) {
      setError(e instanceof Error ? e.message : '讨论启动失败')
    } finally {
      const restored = await refresh() // Recover an accepted POST without issuing a duplicate start.
      if (restored?.session?.session_id === requestedSessionId) {
        accepted = true
        setError(null)
      }
      submittingRef.current = false
      setSubmitting(false)
    }
    return accepted
  }
  const updateNetwork = async (id: string, config: NetworkPatch) => {
    const accepted = await patchDiscussionNetwork(id, config, ...((sessionRef.current?.architecture ?? architecture) === 'distributed' ? ['distributed'] as const : []))
    try {
      const recovered = await getDiscussion(id, ...(architecture === 'distributed' ? [architecture] as const : []))
      setSession(current => current?.session_id === id ? restoreDiscussion(current, recovered) : current)
    } catch { void refresh() } // PATCH was accepted; reconnect/GET repairs projection without retrying it.
    return accepted
  }
  const selectSession = async (id: string) => {
    selectedSessionId.current = id
    setSessionUrl(id)
    setRecovering(true)
    await refresh()
    await refreshEvents()
  }
  const markDraftEdited = () => {
    draftEdited.current = true
  }
  const newExperiment = () => {
    markDraftEdited()
    setMode('auto')
    setNetworkAuto(current => ({ ...current, autoPolicy: 'performance_experiment', selectedKv: 'v0', idealized: true, synthetic: undefined, allowV3: false }))
  }
  const changeArchitecture = (value: Architecture) => {
    if (running || value === architecture) return
    ++refreshVersion.current; ++eventRefreshVersion.current
    selectedSessionId.current = null
    const url = new URL(window.location.href)
    url.searchParams.delete('discussion')
    if (value === 'distributed') url.searchParams.set('architecture', value)
    else url.searchParams.delete('architecture')
    window.history.replaceState(null, '', url)
    setSession(null); setLatest(null); setEvents([]); setLoaded(false); setRecovering(false)
    setError(null); setRecoveryError(null); setNotice(null); setScenario('discussion')
    setNetworkAuto(current => ({ ...current, profileText: '' }))
    draftEdited.current = true
    setArchitecture(value)
  }
  const release = async () => {
    if (!session || running || architecture !== 'distributed') return
    try { await releaseDiscussion(session.session_id); await refresh() }
    catch (e) { setError(e instanceof Error ? e.message : String(e)) }
  }
  return {
    newExperiment, architecture, changeArchitecture, contextTokens, setContextTokens, release,
    modelLoading, changeModel,
    updateNetwork, networkAuto, setNetworkAuto, configurationError, profileProvenance, selectSession,
    hasV2, recomputePolicy, setRecomputePolicy, recomputeValue, setRecomputeValue,
    scenario,
    setScenario,
    secretA,
    setSecretA,
    secretB,
    setSecretB,
    secret,
    question,
    setQuestion,
    rounds,
    setRounds,
    tokens,
    setTokens,
    summarizer,
    setSummarizer,
    mode,
    setMode,
    plan,
    setPlan,
    agentA,
    setAgentA,
    agentB,
    setAgentB,
    preview,
    promptError,
    latest,
    session,
    error,
    notice,
    recoveryError,
    submitting,
    recovering,
    loaded,
    connection,
    running,
    effectiveRounds,
    previewReady,
    valid,
    submit,
    refresh,
    markDraftEdited,
    eventsError,
    refreshEvents,
    events: session
      ? events.filter((e) => e.metadata.session_id === session.session_id && (e.metadata.architecture ?? 'local') === (session.architecture ?? 'local'))
      : [],
  }
}
