export type CommunicationMode = 'TOKEN' | 'KV'
export type ConnectionStatus = 'connected' | 'reconnecting' | 'disconnected'

export interface NetworkState {
  bandwidth_mbps: number
  base_latency_ms: number
}

export interface AgentState {
  agent_id: string
  status: string
}

export interface ModeSelection {
  source: string | null
  target: string | null
  mode: CommunicationMode | null
  estimated_token_latency_ms: number | null
  estimated_kv_latency_ms: number | null
  token_count?: number
  bandwidth_mbps?: number
  base_latency_ms?: number
  cost_boundary?: string
  selector_kind?: string
  selected_mode?: CommunicationMode
  estimated_saving_ms?: number
  estimated_saving_ratio?: number
  token_estimate?: TokenEstimate
  kv_estimate?: KVEstimate
}

export interface TokenEstimate {
  wire_bytes: number
  serialize_ms: number
  link_ms: number
  deserialize_ms: number
  receiver_prefill_ms: number
  total_ms: number
}

export interface KVEstimate {
  raw_kv_bytes: number
  wire_metadata_bytes: number
  wire_bytes: number
  sender_prepare_ms: number
  serialize_ms: number
  link_ms: number
  deserialize_ms: number
  restore_ms: number
  message_ready_total_ms: number
}

export interface AdaptiveExecution {
  source: string
  target: string
  executed_mode: CommunicationMode
  predicted_ms: number
  actual_ms: number
  absolute_error_ms: number
  relative_error: number
  selected_measurement: Record<string, number | boolean>
  sender_text: string
  receiver_text: string
  correctness: boolean | null
}

export interface Communication {
  source: string | null
  target: string | null
  mode: CommunicationMode | null
  payload_bytes: number | null
  transmission_latency_ms: number | null
  estimated_e2e_latency_ms: number | null
}

export interface RuntimeState {
  mode:
    | 'interactive'
    | 'auto'
    | 'observer'
    | 'real_token'
    | 'real_kv'
    | 'real_adaptive'
  status?: 'loading' | 'ready' | 'running' | 'error'
  running: boolean
  next_source: string | null
  next_target: string | null
  active_communication_mode?: CommunicationMode | null
  selector_status?: 'active' | 'not_applicable'
  selector_kind?: string
  cost_boundary?: string
  selector_profile?: {
    profile_name: string
    device_type: string
    model_id: string
    cost_boundary: string
    calibration_token_counts: number[]
    development_calibration: boolean
  }
  model?: string
  device?: string
  max_new_tokens?: number
  warmup_performed?: boolean
  last_error?: string | null
  canonical_handoff_semantics?: boolean
}

export interface RealTokenMetrics {
  source: string | null
  target: string | null
  communication_mode: 'TOKEN'
  measurement_strategy: 'single_run'
  token_count: number | null
  theoretical_token_bytes: number | null
  wire_bytes: number | null
  serialization_ms: number | null
  deserialization_ms: number | null
  receiver_prefill_ms: number | null
  simulated_link_ms: number | null
  token_handoff_cost_ms: number | null
  receiver_decode_mean_ms: number | null
  receiver_decode_median_ms: number | null
  sender_generated_text: string
  receiver_generated_text: string
  token_identity_verified: boolean | null
  protocol_version: number | null
  transport: string
}

export interface RealKVMetrics {
  source: string | null
  target: string | null
  communication_mode: 'KV'
  measurement_strategy: 'single_run'
  canonical_handoff_semantics: boolean | null
  token_count: number | null
  raw_kv_bytes: number | null
  wire_tensor_body_bytes: number | null
  wire_metadata_bytes: number | null
  wire_bytes: number | null
  sender_kv_prepare_ms: number | null
  device_to_host_ms: number | null
  protocol_encode_ms: number | null
  total_serialize_ms: number | null
  local_transfer_ms: number | null
  deserialize_ms: number | null
  cpu_tensor_reconstruction_ms: number | null
  host_to_device_ms: number | null
  cache_construction_ms: number | null
  total_restore_ms: number | null
  receiver_continuation_forward_ms: number | null
  bandwidth_mbps: number | null
  base_latency_ms: number | null
  simulated_link_ms: number | null
  representation_ready_cost_ms: number | null
  message_ready_e2e_cost_ms: number | null
  receiver_prefix_prefill_executed: boolean | null
  sender_generated_text: string
  receiver_generated_text: string
  compatibility_verified: boolean | null
  wire_identity_verified: boolean | null
  canonical_cache_complete: boolean | null
  continuation_verified: boolean | null
  transport: string
  protocol_version: number | null
}

export interface DashboardEvent {
  type: string
  timestamp: string
  source: string | null
  target: string | null
  mode: CommunicationMode | null
  payload_bytes: number | null
  transmission_latency_ms: number | null
  estimated_e2e_latency_ms: number | null
  metadata: Record<string, unknown>
}

export interface DashboardSnapshot {
  system_status: string
  current_task: string | null
  network: NetworkState | null
  agents: Record<string, AgentState>
  last_selection: ModeSelection | null
  last_communication: Communication | null
  real_token_metrics: RealTokenMetrics | null
  real_kv_metrics: RealKVMetrics | null
  adaptive_execution?: AdaptiveExecution | null
  recent_events: DashboardEvent[]
  runtime: RuntimeState
}

export interface EventsResponse {
  events: DashboardEvent[]
  count: number
}

export interface CommunicationControlResponse {
  source: string
  target: string
  communication_mode: CommunicationMode
  selected_mode: CommunicationMode | null
  runtime: RuntimeState
}
