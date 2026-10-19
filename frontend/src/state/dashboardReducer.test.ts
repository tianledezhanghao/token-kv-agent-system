import { describe, expect, it } from 'vitest'

import type { DashboardEvent } from '../types/dashboard'
import { baseSnapshot, tokenSelectionEvent } from '../test/fixtures'
import { dashboardReducer, initialDashboardState } from './dashboardReducer'

describe('dashboardReducer', () => {
  it('hydrates state and safely preserves missing optional values', () => {
    const snapshot = {
      ...baseSnapshot,
      last_selection: null,
      last_communication: null,
    }
    const state = dashboardReducer(initialDashboardState, {
      type: 'hydrate',
      snapshot,
      events: [],
    })

    expect(state.snapshot?.last_selection).toBeNull()
    expect(state.snapshot?.last_communication).toBeNull()
  })

  it('projects TOKEN and KV mode selections from live events', () => {
    let state = dashboardReducer(initialDashboardState, {
      type: 'hydrate',
      snapshot: baseSnapshot,
      events: [],
    })
    state = dashboardReducer(state, {
      type: 'event',
      event: tokenSelectionEvent,
    })
    expect(state.snapshot?.last_selection?.mode).toBe('TOKEN')

    state = dashboardReducer(state, {
      type: 'event',
      event: {
        ...tokenSelectionEvent,
        mode: 'KV',
        source: 'Agent B',
        target: 'Agent A',
        metadata: {
          estimated_token_latency_ms: 25.008,
          estimated_kv_latency_ms: 13,
        },
      },
    })
    expect(state.snapshot?.last_selection).toMatchObject({
      source: 'Agent B',
      target: 'Agent A',
      mode: 'KV',
      estimated_kv_latency_ms: 13,
    })
  })

  it('updates the network from a live backend event', () => {
    const hydrated = dashboardReducer(initialDashboardState, {
      type: 'hydrate',
      snapshot: baseSnapshot,
      events: [],
    })
    const state = dashboardReducer(hydrated, {
      type: 'event',
      event: {
        ...tokenSelectionEvent,
        type: 'NETWORK_STATE_CHANGED',
        source: null,
        target: null,
        mode: null,
        metadata: { bandwidth_mbps: 1000, base_latency_ms: 5 },
      },
    })

    expect(state.snapshot?.network).toEqual({
      bandwidth_mbps: 1000,
      base_latency_ms: 5,
    })
  })

  it('updates the latest communication and agent states from a live event', () => {
    const hydrated = dashboardReducer(initialDashboardState, {
      type: 'hydrate',
      snapshot: baseSnapshot,
      events: [],
    })
    const event: DashboardEvent = {
      ...tokenSelectionEvent,
      type: 'TRANSMISSION_FINISHED',
      mode: 'KV',
      source: 'Agent B',
      target: 'Agent A',
      payload_bytes: 1_000_000,
      transmission_latency_ms: 13,
      estimated_e2e_latency_ms: 13,
      metadata: {},
    }
    const state = dashboardReducer(hydrated, { type: 'event', event })

    expect(state.snapshot?.last_communication).toMatchObject({
      mode: 'KV',
      payload_bytes: 1_000_000,
    })
    expect(state.snapshot?.agents['Agent A'].status).toBe('idle')
    expect(state.events).toContain(event)
  })

  it('projects real Token measurements without creating a selection', () => {
    const hydrated = dashboardReducer(initialDashboardState, {
      type: 'hydrate',
      snapshot: { ...baseSnapshot, last_selection: null },
      events: [],
    })
    const event: DashboardEvent = {
      ...tokenSelectionEvent,
      type: 'REAL_TOKEN_HANDOFF_FINISHED',
      mode: 'TOKEN',
      payload_bytes: 42,
      transmission_latency_ms: 5.0168,
      estimated_e2e_latency_ms: 20.15,
      metadata: {
        token_count: 8,
        theoretical_token_bytes: 18,
        wire_bytes: 42,
        serialization_ms: 0.001,
        deserialization_ms: 0.006,
        receiver_prefill_ms: 15.126,
        simulated_link_ms: 5.0168,
        token_handoff_cost_ms: 20.15,
        sender_generated_text: 'sender',
        receiver_generated_text: 'receiver',
        token_identity_verified: true,
        protocol_version: 1,
        transport: 'InMemoryByteTransport',
      },
    }
    const state = dashboardReducer(hydrated, { type: 'event', event })

    expect(state.snapshot?.last_selection).toBeNull()
    expect(state.snapshot?.real_token_metrics).toMatchObject({
      token_count: 8,
      wire_bytes: 42,
      receiver_prefill_ms: 15.126,
      token_identity_verified: true,
    })
  })

  it('projects real KV measurements without creating a selection', () => {
    const hydrated = dashboardReducer(initialDashboardState, {
      type: 'hydrate',
      snapshot: { ...baseSnapshot, last_selection: null },
      events: [],
    })
    const event: DashboardEvent = {
      ...tokenSelectionEvent,
      type: 'REAL_KV_HANDOFF_FINISHED',
      mode: 'KV',
      payload_bytes: 100_587,
      transmission_latency_ms: 45.2348,
      estimated_e2e_latency_ms: 87.5348,
      metadata: {
        canonical_handoff_semantics: true,
        token_count: 8,
        raw_kv_bytes: 98_304,
        wire_tensor_body_bytes: 98_304,
        wire_metadata_bytes: 2_283,
        wire_bytes: 100_587,
        sender_kv_prepare_ms: 27.2,
        device_to_host_ms: 6.1,
        protocol_encode_ms: 1.6,
        total_serialize_ms: 7.7,
        local_transfer_ms: 0.002,
        deserialize_ms: 0.1,
        cpu_tensor_reconstruction_ms: 2.1,
        host_to_device_ms: 4.2,
        cache_construction_ms: 1,
        total_restore_ms: 7.3,
        receiver_continuation_forward_ms: 14,
        bandwidth_mbps: 20,
        base_latency_ms: 5,
        simulated_link_ms: 45.2348,
        representation_ready_cost_ms: 60.3348,
        message_ready_e2e_cost_ms: 87.5348,
        receiver_prefix_prefill_executed: false,
        sender_generated_text: 'sender',
        receiver_generated_text: 'receiver',
        compatibility_verified: true,
        wire_identity_verified: true,
        canonical_cache_complete: true,
        continuation_verified: true,
        protocol_version: 1,
        transport: 'InMemoryByteTransport',
      },
    }
    const state = dashboardReducer(hydrated, { type: 'event', event })

    expect(state.snapshot?.last_selection).toBeNull()
    expect(state.snapshot?.real_kv_metrics).toMatchObject({
      token_count: 8,
      raw_kv_bytes: 98_304,
      wire_bytes: 100_587,
      representation_ready_cost_ms: 60.3348,
      message_ready_e2e_cost_ms: 87.5348,
      receiver_prefix_prefill_executed: false,
      continuation_verified: true,
    })
  })

  it('projects real adaptive selection before actual execution arrives', () => {
    const hydrated = dashboardReducer(initialDashboardState, {
      type: 'hydrate',
      snapshot: {
        ...baseSnapshot,
        last_selection: null,
        runtime: {
          ...baseSnapshot.runtime,
          mode: 'real_adaptive',
          active_communication_mode: null,
        },
      },
      events: [],
    })
    const selection: DashboardEvent = {
      ...tokenSelectionEvent,
      type: 'REAL_SELECTION_MADE',
      mode: 'KV',
      metadata: {
        token_count: 8,
        bandwidth_mbps: 1000,
        base_latency_ms: 5,
        cost_boundary: 'MESSAGE_READY',
        selector_kind: 'real_cost_based',
        selected_mode: 'KV',
        estimated_saving_ms: 4,
        estimated_saving_ratio: 0.1,
        token_estimated_ms: 25,
        kv_estimated_ms: 21,
        token_estimate: { wire_bytes: 42, total_ms: 25 },
        kv_estimate: { wire_bytes: 100587, message_ready_total_ms: 21 },
      },
    }
    const selected = dashboardReducer(hydrated, {
      type: 'event',
      event: selection,
    })
    expect(selected.snapshot?.runtime.active_communication_mode).toBe('KV')
    expect(selected.snapshot?.last_selection?.selected_mode).toBe('KV')
    expect(selected.snapshot?.adaptive_execution).toBeNull()
    const finished = dashboardReducer(selected, {
      type: 'event',
      event: {
        ...selection,
        type: 'REAL_ADAPTIVE_HANDOFF_FINISHED',
        metadata: {
          actually_executed_mode: 'KV',
          predicted_selected_cost_ms: 21,
          actual_selected_cost_ms: 23,
          absolute_error_ms: 2,
          relative_error: 2 / 21,
          selected_measurement: { wire_bytes: 100587, link_ms: 5.8 },
          sender_text: 'sender',
          receiver_text: 'receiver',
          correctness: true,
        },
      },
    })
    expect(finished.snapshot?.adaptive_execution?.executed_mode).toBe('KV')
    expect(finished.snapshot?.adaptive_execution?.absolute_error_ms).toBe(2)
  })
})

describe('Missing real measurements', () => {
  it('does not turn missing stage timings into zero or identity into failure', () => {
    const state = dashboardReducer(initialDashboardState, {
      type: 'hydrate',
      snapshot: baseSnapshot,
      events: [],
    })
    const next = dashboardReducer(state, {
      type: 'event',
      event: {
        ...tokenSelectionEvent,
        type: 'REAL_TOKEN_HANDOFF_FINISHED',
        metadata: {},
      },
    })
    expect(next.snapshot?.real_token_metrics?.serialization_ms).toBeNull()
    expect(
      next.snapshot?.real_token_metrics?.token_identity_verified,
    ).toBeNull()
  })
  it('does not crash on an incomplete adaptive estimate event', () => {
    const state = dashboardReducer(initialDashboardState, {
      type: 'hydrate',
      snapshot: baseSnapshot,
      events: [],
    })
    const next = dashboardReducer(state, {
      type: 'event',
      event: {
        ...tokenSelectionEvent,
        type: 'REAL_SELECTION_MADE',
        metadata: { selected_mode: 'TOKEN' },
      },
    })
    expect(next.snapshot?.last_selection?.estimated_token_latency_ms).toBeNull()
  })
})

describe('Concurrent control and snapshot recovery', () => {
  it('unlocks a runtime hydrated as running when another client finishes', () => {
    const state = dashboardReducer(initialDashboardState, {
      type: 'hydrate',
      snapshot: {
        ...baseSnapshot,
        runtime: { ...baseSnapshot.runtime, running: true },
      },
      events: [],
    })
    const finished = dashboardReducer(state, {
      type: 'event',
      event: { ...tokenSelectionEvent, type: 'TRANSMISSION_FINISHED' },
    })
    expect(finished.snapshot?.runtime.running).toBe(false)
  })
  it('clears the prior adaptive selection at the next run start', () => {
    const state = dashboardReducer(initialDashboardState, {
      type: 'hydrate',
      snapshot: {
        ...baseSnapshot,
        runtime: { ...baseSnapshot.runtime, active_communication_mode: 'KV' },
      },
      events: [],
    })
    const started = dashboardReducer(state, {
      type: 'event',
      event: { ...tokenSelectionEvent, type: 'REAL_ADAPTIVE_HANDOFF_STARTED' },
    })
    expect(started.snapshot?.runtime.running).toBe(true)
    expect(started.snapshot?.runtime.active_communication_mode).toBeNull()
    expect(started.snapshot?.last_selection).toBeNull()
  })
  it('replays frames received while a REST snapshot is pending without duplicating logs', () => {
    const received = {
      ...tokenSelectionEvent,
      type: 'NETWORK_STATE_CHANGED',
      timestamp: '2026-10-14T05:00:00Z',
      metadata: { bandwidth_mbps: 1000, base_latency_ms: 5 },
    }
    const snapshot = { ...baseSnapshot, recent_events: [] }
    const state = dashboardReducer(initialDashboardState, {
      type: 'hydrate',
      snapshot,
      events: [received],
      pendingEvents: [received],
    })
    expect(state.snapshot?.network?.bandwidth_mbps).toBe(1000)
    expect(state.events).toHaveLength(1)
  })
})
