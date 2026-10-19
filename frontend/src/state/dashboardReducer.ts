import type {
  CommunicationMode,
  TokenEstimate,
  KVEstimate,
  DashboardEvent,
  DashboardSnapshot,
  NetworkState,
} from '../types/dashboard'

export interface DashboardViewState {
  snapshot: DashboardSnapshot | null
  events: DashboardEvent[]
}

export type DashboardAction =
  | {
      type: 'hydrate'
      snapshot: DashboardSnapshot
      events: DashboardEvent[]
      pendingEvents?: DashboardEvent[]
    }
  | { type: 'event'; event: DashboardEvent }

export const initialDashboardState: DashboardViewState = {
  snapshot: null,
  events: [],
}

const eventLimit = 100

function numberMetadata(event: DashboardEvent, key: string): number | null {
  const value = event.metadata[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function setAgent(
  agents: DashboardSnapshot['agents'],
  agentId: string | null,
  status: string,
): DashboardSnapshot['agents'] {
  if (!agentId) return agents
  return { ...agents, [agentId]: { agent_id: agentId, status } }
}

function projectEvent(
  snapshot: DashboardSnapshot,
  event: DashboardEvent,
): DashboardSnapshot {
  if (
    [
      'AGENT_GENERATION_STARTED',
      'MODE_SELECTED',
      'TRANSMISSION_STARTED',
      'REAL_TOKEN_HANDOFF_STARTED',
      'REAL_KV_HANDOFF_STARTED',
      'REAL_ADAPTIVE_HANDOFF_STARTED',
    ].includes(event.type)
  )
    snapshot = {
      ...snapshot,
      runtime: { ...snapshot.runtime, running: true, status: 'running' },
    }
  if (
    [
      'TRANSMISSION_FINISHED',
      'REAL_TOKEN_HANDOFF_FINISHED',
      'REAL_KV_HANDOFF_FINISHED',
      'REAL_ADAPTIVE_HANDOFF_FINISHED',
      'REAL_TOKEN_HANDOFF_FAILED',
      'REAL_KV_HANDOFF_FAILED',
      'REAL_ADAPTIVE_HANDOFF_FAILED',
    ].includes(event.type)
  )
    snapshot = {
      ...snapshot,
      runtime: {
        ...snapshot.runtime,
        running: false,
        status: event.type.endsWith('FAILED') ? 'error' : 'ready',
        last_error: event.type.endsWith('FAILED')
          ? String(event.metadata.error ?? '通信失败')
          : null,
      },
    }
  let agents = snapshot.agents

  switch (event.type) {
    case 'NETWORK_STATE_CHANGED': {
      const bandwidth = numberMetadata(event, 'bandwidth_mbps')
      const baseLatency = numberMetadata(event, 'base_latency_ms')
      const network: NetworkState | null =
        bandwidth !== null && baseLatency !== null
          ? { bandwidth_mbps: bandwidth, base_latency_ms: baseLatency }
          : snapshot.network
      return { ...snapshot, network }
    }
    case 'AGENT_GENERATION_STARTED': {
      agents = setAgent(agents, event.source, 'generating')
      const prompt = event.metadata.prompt
      return {
        ...snapshot,
        agents,
        system_status: 'running',
        current_task: typeof prompt === 'string' ? prompt : null,
        last_selection: null,
      }
    }
    case 'AGENT_GENERATION_FINISHED':
      return { ...snapshot, agents: setAgent(agents, event.source, 'idle') }
    case 'MODE_SELECTED':
      agents = setAgent(agents, event.source, 'sending')
      agents = setAgent(agents, event.target, 'waiting')
      return {
        ...snapshot,
        agents,
        last_selection: {
          source: event.source,
          target: event.target,
          mode: event.mode,
          estimated_token_latency_ms: numberMetadata(
            event,
            'estimated_token_latency_ms',
          ),
          estimated_kv_latency_ms: numberMetadata(
            event,
            'estimated_kv_latency_ms',
          ),
        },
      }
    case 'REAL_ADAPTIVE_HANDOFF_STARTED':
      agents = setAgent(agents, event.source, 'generating')
      agents = setAgent(agents, event.target, 'waiting')
      return {
        ...snapshot,
        agents,
        system_status: 'running',
        current_task: 'Real Adaptive handoff',
        last_selection: null,
        adaptive_execution: null,
        runtime: { ...snapshot.runtime, active_communication_mode: null },
      }
    case 'REAL_SELECTION_MADE': {
      const selected = event.metadata.selected_mode as CommunicationMode
      agents = setAgent(agents, event.source, 'sending')
      return {
        ...snapshot,
        agents,
        adaptive_execution: null,
        runtime: { ...snapshot.runtime, active_communication_mode: selected },
        last_selection: {
          source: event.source,
          target: event.target,
          mode: selected,
          estimated_token_latency_ms:
            numberMetadata(event, 'token_estimated_ms') ??
            (event.metadata.token_estimate as TokenEstimate | undefined)
              ?.total_ms ??
            null,
          estimated_kv_latency_ms:
            numberMetadata(event, 'kv_estimated_ms') ??
            (event.metadata.kv_estimate as KVEstimate | undefined)
              ?.message_ready_total_ms ??
            null,
          token_count: numberMetadata(event, 'token_count') ?? undefined,
          bandwidth_mbps: numberMetadata(event, 'bandwidth_mbps') ?? undefined,
          base_latency_ms:
            numberMetadata(event, 'base_latency_ms') ?? undefined,
          cost_boundary: String(event.metadata.cost_boundary),
          selector_kind: String(event.metadata.selector_kind),
          selected_mode: selected,
          estimated_saving_ms:
            numberMetadata(event, 'estimated_saving_ms') ?? undefined,
          estimated_saving_ratio:
            numberMetadata(event, 'estimated_saving_ratio') ?? undefined,
          token_estimate: event.metadata.token_estimate as TokenEstimate,
          kv_estimate: event.metadata.kv_estimate as KVEstimate,
        },
      }
    }
    case 'SELECTED_PATH_STARTED':
      return {
        ...snapshot,
        agents: setAgent(agents, event.target, 'receiving'),
      }
    case 'REAL_ADAPTIVE_HANDOFF_FINISHED': {
      const data = event.metadata
      const measurement = data.selected_measurement as Record<
        string,
        number | boolean
      >
      agents = setAgent(agents, event.source, 'idle')
      agents = setAgent(agents, event.target, 'idle')
      return {
        ...snapshot,
        agents,
        system_status: 'idle',
        current_task: null,
        adaptive_execution: {
          source: event.source ?? '',
          target: event.target ?? '',
          executed_mode: data.actually_executed_mode as CommunicationMode,
          predicted_ms: data.predicted_selected_cost_ms as number,
          actual_ms: data.actual_selected_cost_ms as number,
          absolute_error_ms: data.absolute_error_ms as number,
          relative_error: data.relative_error as number,
          selected_measurement: measurement,
          sender_text: String(data.sender_text),
          receiver_text: String(data.receiver_text),
          correctness:
            typeof data.correctness === 'boolean' ? data.correctness : null,
        },
        last_communication: {
          source: event.source,
          target: event.target,
          mode: data.actually_executed_mode as CommunicationMode,
          payload_bytes: measurement.wire_bytes as number,
          transmission_latency_ms: measurement.link_ms as number,
          estimated_e2e_latency_ms: data.predicted_selected_cost_ms as number,
        },
      }
    }
    case 'REAL_ADAPTIVE_HANDOFF_FAILED':
      agents = setAgent(agents, event.source, 'idle')
      agents = setAgent(agents, event.target, 'idle')
      return {
        ...snapshot,
        agents,
        system_status: 'error',
        current_task: null,
        runtime: {
          ...snapshot.runtime,
          active_communication_mode:
            (event.metadata
              .active_communication_mode as CommunicationMode | null) ?? null,
        },
      }
    case 'TRANSMISSION_STARTED':
      agents = setAgent(agents, event.source, 'sending')
      agents = setAgent(agents, event.target, 'receiving')
      return { ...snapshot, agents }
    case 'TRANSMISSION_FINISHED':
      agents = setAgent(agents, event.source, 'idle')
      agents = setAgent(agents, event.target, 'idle')
      return {
        ...snapshot,
        agents,
        system_status: 'idle',
        current_task: null,
        last_communication: {
          source: event.source,
          target: event.target,
          mode: event.mode,
          payload_bytes: event.payload_bytes,
          transmission_latency_ms: event.transmission_latency_ms,
          estimated_e2e_latency_ms: event.estimated_e2e_latency_ms,
        },
      }
    case 'REAL_TOKEN_HANDOFF_STARTED':
      agents = setAgent(agents, event.source, 'sending')
      agents = setAgent(agents, event.target, 'waiting')
      return {
        ...snapshot,
        agents,
        system_status: 'running',
        current_task: 'Real Token direct causal continuation',
      }
    case 'TOKEN_TRANSFERRED':
      agents = setAgent(agents, event.source, 'sending')
      agents = setAgent(agents, event.target, 'receiving')
      return { ...snapshot, agents }
    case 'RECEIVER_PREFILL_FINISHED':
      return {
        ...snapshot,
        agents: setAgent(agents, event.target, 'generating'),
      }
    case 'REAL_TOKEN_HANDOFF_FINISHED': {
      agents = setAgent(agents, event.source, 'idle')
      agents = setAgent(agents, event.target, 'idle')
      const number = (key: string) => numberMetadata(event, key)
      return {
        ...snapshot,
        agents,
        system_status: 'idle',
        current_task: null,
        last_communication: {
          source: event.source,
          target: event.target,
          mode: 'TOKEN',
          payload_bytes: event.payload_bytes,
          transmission_latency_ms: event.transmission_latency_ms,
          estimated_e2e_latency_ms: event.estimated_e2e_latency_ms,
        },
        real_token_metrics: {
          source: event.source,
          target: event.target,
          communication_mode: 'TOKEN',
          measurement_strategy: 'single_run',
          token_count: number('token_count'),
          theoretical_token_bytes: number('theoretical_token_bytes'),
          wire_bytes: number('wire_bytes'),
          serialization_ms: number('serialization_ms'),
          deserialization_ms: number('deserialization_ms'),
          receiver_prefill_ms: number('receiver_prefill_ms'),
          simulated_link_ms: number('simulated_link_ms'),
          token_handoff_cost_ms: number('token_handoff_cost_ms'),
          receiver_decode_mean_ms: numberMetadata(
            event,
            'receiver_decode_mean_ms',
          ),
          receiver_decode_median_ms: numberMetadata(
            event,
            'receiver_decode_median_ms',
          ),
          sender_generated_text: String(
            event.metadata.sender_generated_text ?? '',
          ),
          receiver_generated_text: String(
            event.metadata.receiver_generated_text ?? '',
          ),
          token_identity_verified:
            typeof event.metadata.token_identity_verified === 'boolean'
              ? event.metadata.token_identity_verified
              : null,
          protocol_version: number('protocol_version'),
          transport: String(event.metadata.transport ?? ''),
        },
      }
    }
    case 'REAL_TOKEN_HANDOFF_FAILED':
      agents = setAgent(agents, event.source, 'idle')
      agents = setAgent(agents, event.target, 'idle')
      return { ...snapshot, agents, system_status: 'error', current_task: null }
    case 'REAL_KV_HANDOFF_STARTED':
      agents = setAgent(agents, event.source, 'sending')
      agents = setAgent(agents, event.target, 'waiting')
      return {
        ...snapshot,
        agents,
        system_status: 'running',
        current_task: 'Real KV canonical handoff',
      }
    case 'KV_TRANSFERRED':
      agents = setAgent(agents, event.source, 'sending')
      agents = setAgent(agents, event.target, 'receiving')
      return { ...snapshot, agents }
    case 'KV_RESTORED':
      return {
        ...snapshot,
        agents: setAgent(agents, event.target, 'generating'),
      }
    case 'REAL_KV_HANDOFF_FINISHED': {
      agents = setAgent(agents, event.source, 'idle')
      agents = setAgent(agents, event.target, 'idle')
      const number = (key: string) => numberMetadata(event, key)
      return {
        ...snapshot,
        agents,
        system_status: 'idle',
        current_task: null,
        last_communication: {
          source: event.source,
          target: event.target,
          mode: 'KV',
          payload_bytes: event.payload_bytes,
          transmission_latency_ms: event.transmission_latency_ms,
          estimated_e2e_latency_ms: event.estimated_e2e_latency_ms,
        },
        real_kv_metrics: {
          source: event.source,
          target: event.target,
          communication_mode: 'KV',
          measurement_strategy: 'single_run',
          canonical_handoff_semantics:
            typeof event.metadata.canonical_handoff_semantics === 'boolean'
              ? event.metadata.canonical_handoff_semantics
              : null,
          token_count: number('token_count'),
          raw_kv_bytes: number('raw_kv_bytes'),
          wire_tensor_body_bytes: number('wire_tensor_body_bytes'),
          wire_metadata_bytes: number('wire_metadata_bytes'),
          wire_bytes: number('wire_bytes'),
          sender_kv_prepare_ms: number('sender_kv_prepare_ms'),
          device_to_host_ms: number('device_to_host_ms'),
          protocol_encode_ms: number('protocol_encode_ms'),
          total_serialize_ms: number('total_serialize_ms'),
          local_transfer_ms: number('local_transfer_ms'),
          deserialize_ms: number('deserialize_ms'),
          cpu_tensor_reconstruction_ms: number('cpu_tensor_reconstruction_ms'),
          host_to_device_ms: number('host_to_device_ms'),
          cache_construction_ms: number('cache_construction_ms'),
          total_restore_ms: number('total_restore_ms'),
          receiver_continuation_forward_ms: number(
            'receiver_continuation_forward_ms',
          ),
          bandwidth_mbps: number('bandwidth_mbps'),
          base_latency_ms: number('base_latency_ms'),
          simulated_link_ms: number('simulated_link_ms'),
          representation_ready_cost_ms: number('representation_ready_cost_ms'),
          message_ready_e2e_cost_ms: number('message_ready_e2e_cost_ms'),
          receiver_prefix_prefill_executed:
            typeof event.metadata.receiver_prefix_prefill_executed === 'boolean'
              ? event.metadata.receiver_prefix_prefill_executed
              : null,
          sender_generated_text: String(
            event.metadata.sender_generated_text ?? '',
          ),
          receiver_generated_text: String(
            event.metadata.receiver_generated_text ?? '',
          ),
          compatibility_verified:
            typeof event.metadata.compatibility_verified === 'boolean'
              ? event.metadata.compatibility_verified
              : null,
          wire_identity_verified:
            typeof event.metadata.wire_identity_verified === 'boolean'
              ? event.metadata.wire_identity_verified
              : null,
          canonical_cache_complete:
            typeof event.metadata.canonical_cache_complete === 'boolean'
              ? event.metadata.canonical_cache_complete
              : null,
          continuation_verified:
            typeof event.metadata.continuation_verified === 'boolean'
              ? event.metadata.continuation_verified
              : null,
          transport: String(event.metadata.transport ?? ''),
          protocol_version: number('protocol_version'),
        },
      }
    }
    case 'REAL_KV_HANDOFF_FAILED':
      agents = setAgent(agents, event.source, 'idle')
      agents = setAgent(agents, event.target, 'idle')
      return { ...snapshot, agents, system_status: 'error', current_task: null }
    default:
      return snapshot
  }
}

export function dashboardReducer(
  state: DashboardViewState,
  action: DashboardAction,
): DashboardViewState {
  if (action.type === 'hydrate') {
    const pending = action.pendingEvents ?? []
    const known = new Set(
      action.snapshot.recent_events.map((e) => JSON.stringify(e)),
    )
    const watermark = action.snapshot.recent_events.at(-1)?.timestamp ?? ''
    const replay = pending.filter(
      (e) => !known.has(JSON.stringify(e)) && e.timestamp >= watermark,
    )
    const unique = new Map(
      [...action.events, ...pending].map((e) => [JSON.stringify(e), e]),
    )
    return {
      snapshot: replay.reduce(projectEvent, action.snapshot),
      events: [...unique.values()]
        .sort((a, b) => a.timestamp.localeCompare(b.timestamp))
        .slice(-eventLimit),
    }
  }
  if (!state.snapshot) {
    return {
      ...state,
      events: [...state.events, action.event].slice(-eventLimit),
    }
  }
  return {
    snapshot: projectEvent(state.snapshot, action.event),
    events: [...state.events, action.event].slice(-eventLimit),
  }
}
