export interface SyntheticAssumptions { schema: 'synthetic_compute_v1'; source: 'synthetic_assumption'; units: 'ms'; a: number; b: number; c: number; d: number; e: number; f: number }
export type Architecture = 'local' | 'distributed'
export type ManualMode = 'token' | 'kv:v0' | 'kv:v1' | 'kv:v2' | 'kv:v3'
export interface CostProfile {
  version: string; provenance: 'measured_incremental' | 'synthetic/test_only' | 'measured'
  model_identity: Record<string, unknown>; stages?: Record<string, Record<string, number>>
  schema?: string; environment?: Record<string, unknown>; points?: Record<string, unknown>[]
  calibrated_at?: string; evidence_sha256?: string; units?: string; reliability?: string; relative_error_floor?: number; startup_cost_ms?: number | null; startup_uncertainty_ms?: number | null
  sample_count: number; message_token_range?: number[]; cache_token_range?: number[]
  recompute_token_range?: number[]; notes: string; semantics?: string
}
export interface ShannonChannel {
  bandwidth_mhz: number; snr_mean_db: number; snr_sigma_db: number; correlation: number; seed: number; efficiency: number
  snr_min_db?: number | null; snr_max_db?: number | null
}
export interface ChannelSample extends ShannonChannel { snr_db: number; capacity_mbps: number; effective_rate_mbps: number; clipped?: boolean }
export interface NetworkPatch { bandwidth_mbps?: number; base_latency_ms?: number; manual_override?: boolean; channel?: Partial<ShannonChannel> }
export interface NetworkSnapshot {
  channel?: ChannelSample | null
  bandwidth_mbps: number; base_latency_ms: number; source: string
  handoff_index?: number; network_mode?: 'manual' | 'trace' | 'shannon'; simulation_step?: number
  config_revision?: number; trace_exhausted?: boolean; trace_id?: string; is_simulated?: boolean
}
export interface NetworkState {
  config: { channel?: ShannonChannel | null; mode: 'manual' | 'trace' | 'shannon'; bandwidth_mbps: number; base_latency_ms: number; source: string; trace_mbps: number[] }
  config_revision: number; next_unfrozen_handoff_index: number; is_simulated: boolean; effective_rule: string
}
export interface NetworkUpdate {
  before: NetworkState; after: NetworkState; config_revision: number; effective_from_handoff_index: number
  accepted_at: string; source: string; mode_before: string; mode_after: string
}
export interface CandidateEstimate {
  candidate: string; predicted_wire_bytes: number; predicted_local_ms: number | null
  simulated_transfer_ms: number | null; predicted_total_ms: number | null; cost_status: string
  uncertainty_ms?: number; cost_source?: string; calibration_source?: string; prediction_method?: string; reliability?: string
  predicted_stage_ms?: Record<string, number>; recomputed_tokens?: number
}
export interface AutoDecision {
  synthetic_assumptions?: SyntheticAssumptions | null
  crossover?: { status: string; cost_source?: string; effective_rate_mbps?: number; snr_threshold_db?: number; below?: string; above?: string; preferred?: string; reason?: string; within_mean_plus_minus_2sigma?: boolean; within_configured_clip_bounds?: boolean };
  auto_policy: string; selected_candidate: string; selection_reason: string; profile_provenance: string; quality_risk: boolean
  profile_evidence_sha256?: string | null; profile_calibrated_at?: string | null; profile_sample_count?: number | null; prediction_method?: string; reliability?: string
  candidate_set?: string[]; selected_kv?: string; idealized?: boolean
  candidate_estimates?: CandidateEstimate[]; excluded_candidates?: { candidate: string; reason: string; error?: string }[]
  network?: NetworkSnapshot; profile_notes?: string; prediction_config_version?: string; decision_status?: string
}
export interface DecisionRecord extends Partial<AutoDecision> {
  handoff_index?: number; sequence?: number; message_id?: string; source?: string; agent?: string; target?: string
  error?: string; decision_status?: string
}
export interface ActiveHandoff {
  sequence: number; message_id?: string; agent: string; target: string
  network?: NetworkSnapshot; decision?: AutoDecision; auto_decision?: AutoDecision | null
  requested_mode?: string; selected_mode?: string; selected_kv_strategy?: string | null
  decision_status?: string; error?: string
}
export interface DiscussionConfig {
  synthetic_assumptions?: SyntheticAssumptions | null
  channel?: ShannonChannel; advanced_rounds?: boolean
  architecture?: Architecture; max_context_tokens?: number
  auto_policy?: 'quality_first' | 'performance_experiment'; auto_candidates?: ManualMode[]
  auto_kv_version?: 'v0' | 'v1' | 'v2' | 'v3'; idealized_simulation?: boolean
  allow_experimental_v3?: boolean; cost_profile?: CostProfile | null
  network_mode?: 'manual' | 'trace' | 'shannon'; bandwidth_mbps?: number; base_latency_ms?: number; bandwidth_trace?: number[]
  scenario?: 'discussion' | 'secret_sum'; secret_a?: number; secret_b?: number
  kv_recompute_tokens?: number | null; kv_recompute_ratio?: number | null
  question: string; rounds: number; max_new_tokens: number; summarizer: 'A' | 'B'
  mode: ManualMode | 'mixed' | 'auto'
  handoff_plan: { round: number; source: 'A' | 'B'; target: 'A' | 'B'; mode: ManualMode }[]
  agent_a_system_prompt?: string | null; agent_b_system_prompt?: string | null
}
export interface PromptConfig {
  agent_a: { text: string; source: 'default' | 'custom' }
  agent_b: { text: string; source: 'default' | 'custom' }
  version: string
  stages: { a_first: string; b_first: string; a_followup: string; b_followup: string; final: string }
}
export interface Turn {
  sequence: number; round: number; agent: string; target?: string; text: string; token_count: number
  configured_max_new_tokens?: number | null; effective_max_new_tokens?: number | null
  context_remaining_tokens?: number | null; budget_limit_reason?: string | null
  termination_reason: string; message_id: string; received_message_id?: string | null
}
export interface ReceiverReceipt {
  schema: string; session_id: string; message_id: string; handoff: number; source: string; target: string
  mode: string; kv_strategy: string | null; status: string; state_committed: boolean; cache_imported: boolean
  stages?: string[]; packet_sha256: string; receiver_boot_id: string
}
export interface WorkerStatus {
  status: string; agent_id?: string; pid?: number; boot_id?: string; model?: { model_name: string; device: string; dtype: string; revision?: string }
  error?: string
}
export interface DistributedHealth {
  measurement_source?: string; status: string; available: boolean; coordinator: string; workers: Record<string, WorkerStatus>
  relays: Record<string, { status: string; pid?: number; check: string; errors: string[] }>
  model_contract_consistent: boolean | null; startup: string
}
export interface Handoff {
  architecture?: Architecture; session_id?: string; actual_execution_status?: string
  receiver_receipt?: ReceiverReceipt | null; delivery_identity?: Record<string, unknown> | null
  socket_measurement?: { sent_bytes: number; received_bytes: number; outcome: string } | null
  protocol_bytes?: number | null; application_bytes?: number | null; measurement_source?: string
  observed_handoff_wall_ms?: number | null; socket_application_delivery_ms?: number | null; receiver_processing_ms?: number | null
  sequence: number; source: string; target: string; discussion_round: number
  message_id?: string | null
  network?: NetworkSnapshot
  auto_decision?: AutoDecision | null
  observed_stage_ms?: Record<string, number> | null; observation_scope?: string
  selected_mode?: string | null; selected_kv_strategy?: string | null
  wire_prediction_error_bytes?: number | null;
  predicted_wire_bytes?: number | null; predicted_local_ms?: number | null
  predicted_total_ms?: number | null; observed_local_ms?: number | null; simulated_transfer_ms?: number | null
  requested_mode: string; executed_mode: string | null
  requested_kv_strategy: string | null; executed_kv_strategy: string | null
  message_content_token_count: number; wire_bytes: number
  sender_kv_span: number[]; receiver_kv_target_span: number[] | null
  receiver_cache_length_before: number; receiver_cache_length_after: number
  receiver_message_prefill_count: number | null; kv_repair_method: string | null
  receiver_recomputed_token_count?: number | null; recompute_forward_count?: number | null
  repaired_token_indices?: number[]; repaired_layer_indices?: number[]; repair_layer_token_count?: number
  rope_position_delta: number | null; error: string | null; fallback_reason: string | null
  append_succeeded: boolean | null; unpack_succeeded: boolean | null; cache_ledger_consistent: boolean | null
}
export interface Summary {
  rounds: number; completed_turns: number; handoff_count: number; attempted_handoff_count: number
  kv_v2_handoffs?: number; kv_v3_handoffs?: number; repair_layer_token_count?: number; recomputed_tokens?: number
  token_handoffs: number; kv_v0_handoffs: number; kv_v1_handoffs: number; total_wire_bytes: number
  final_cache_lengths: Record<string, number>
}
export interface DiscussionSession {
  session_id: string; status: 'running' | 'completed' | 'failed' | 'delivery_unknown'; architecture?: Architecture; workers?: Record<string, WorkerStatus>; resumable?: boolean; revision: number
  config: DiscussionConfig; phase: string; round: number; agent: string | null; sequence: number
  prompt_config: PromptConfig
  score?: { engineering_success: boolean; a_sent_own_secret: boolean | null; b_sent_own_secret: boolean | null; b_received_a: boolean | null; a_received_b: boolean | null; information_exchange_success: boolean | null; final_sum_correct: boolean | null; task_success: boolean | null; expected_sum: number; note: string } | null
  model: LatestDiscussion['model']
  network?: NetworkState; network_updates?: NetworkUpdate[]
  auto_decisions?: DecisionRecord[]; active_handoff?: ActiveHandoff | null
  turns: Turn[]; handoffs: Handoff[]; final_answer: Omit<Turn, 'sequence' | 'round'> | null
  summary: Summary | null; error: string | null; failure_stage: string | null
  termination_reason: string | null; cache_lengths: Record<string, number>; cache_released: boolean
  transport: string; created_at: string; updated_at: string
}
export interface LatestDiscussion {
  architecture?: Architecture; health?: DistributedHealth; profile_status?: string
  model_switch?: { loading: boolean; error: string | null; options: string[]; worker_pid?: number | null }
  active_session_id?: string | null
  recent_sessions?: { architecture?: Architecture; session_id: string; status: DiscussionSession['status']; question: string; mode: string; created_at: string }[]
  cost_model_identity?: Record<string, unknown> | null
  available: boolean; session: DiscussionSession | null
  model: { model_name: string; device: string; dtype: string } | null
}
