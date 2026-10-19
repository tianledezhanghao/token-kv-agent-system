import type { CostProfile, SyntheticAssumptions, DiscussionConfig, ManualMode, ShannonChannel } from './types'

export const candidates: ManualMode[] = ['token', 'kv:v0', 'kv:v1', 'kv:v2', 'kv:v3']
export const defaultSynthetic: SyntheticAssumptions = { schema: 'synthetic_compute_v1', source: 'synthetic_assumption', units: 'ms', a: 20, b: 2, c: 0.002, d: 5, e: 0.05, f: 0.2 }
export const coefficientNames = ['a','b','c','d','e','f'] as const
export type SyntheticDraft = Record<typeof coefficientNames[number], string>
export const syntheticDraft = (c = defaultSynthetic): SyntheticDraft => Object.fromEntries(coefficientNames.map(k => [k, String(c[k])])) as SyntheticDraft
export function parseSynthetic(d = syntheticDraft()): SyntheticAssumptions {
  const result = { ...defaultSynthetic }
  for (const k of coefficientNames) {
    const text = d[k], value = Number(text)
    if (!text.trim() || !/^[+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$/.test(text.trim()) || !Number.isFinite(value) || value < 0 || value > 1e9) throw new Error(`假设系数 ${k} 必须是 0–1e9 的有限非负数。`)
    result[k] = value
  }
  return result
}
export interface NetworkAutoDraft {
  channel: ShannonChannel; networkMode: 'manual' | 'trace' | 'shannon'; bandwidth: string; latency: string; trace: string
  autoPolicy: 'quality_first' | 'performance_experiment'; selectedKv: 'v0' | 'v1' | 'v2' | 'v3'; idealized: boolean
  synthetic?: SyntheticDraft
  allowV3: boolean; profileText: string
}
export const defaultChannel: ShannonChannel = { bandwidth_mhz: 20, snr_mean_db: 10, snr_sigma_db: 4, correlation: 0.85, seed: 42, efficiency: 0.8 }
export function validateChannel(c: ShannonChannel) {
  for (const [name, low, high] of [['bandwidth_mhz', .001, 100000], ['snr_mean_db', -120, 120], ['snr_sigma_db', 0, 40], ['correlation', 0, 1], ['efficiency', 0, 1]] as const) {
    if (!Number.isFinite(c[name]) || c[name] < low || c[name] > high) throw new Error(`${name} 必须在 ${low}–${high} 范围内。`)
  }
  if (c.correlation === 1 || c.efficiency === 0 || !Number.isInteger(c.seed) || c.seed < 0 || c.seed > 4294967295) throw new Error('ρ 必须小于1，η 必须大于0，seed 为 uint32 整数。')
  for (const x of [c.snr_min_db,c.snr_max_db]) if (x != null && (!Number.isFinite(x) || x < -120 || x > 120)) throw new Error('SNR 边界必须在 −120–120 dB。')
  if (c.snr_min_db != null && c.snr_max_db != null && c.snr_min_db > c.snr_max_db) throw new Error('SNR 下界不能大于上界。')
  return c
}
export const defaultNetworkAuto: NetworkAutoDraft = {
  channel: defaultChannel, networkMode: 'shannon', bandwidth: '20', latency: '0', trace: '20,1000,50,500',
  autoPolicy: 'performance_experiment', selectedKv: 'v0', idealized: true, allowV3: false, profileText: '',
}
const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v)
const stageNames = ['token_handoff', 'kv_base', 'rope', 'suffix', 'selection', 'slot', 'kv_contract_startup']
const costNames = ['fixed_ms', 'per_message_token_ms', 'per_cache_token_ms', 'per_recomputed_token_ms']
const profileNames = ['version', 'provenance', 'model_identity', 'stages', 'sample_count', 'message_token_range', 'cache_token_range', 'notes', 'semantics', 'recompute_token_range', 'workload_domain']
export function parseProfile(text: string, identity?: Record<string, unknown> | null): CostProfile | null {
  if (!text.trim()) return null
  let p: unknown
  try { p = JSON.parse(text) } catch { throw new Error('成本 Profile 必须是合法 JSON 对象。') }
  if (object(p) && p.schema === 'generalized_incremental_cost_v1') {
    const names = ['schema','version','provenance','model_identity','environment','points','sample_count','calibrated_at','evidence_sha256','notes','units','reliability','relative_error_floor','startup_cost_ms','startup_uncertainty_ms']
    if (Object.keys(p).some(k => !names.includes(k)) || p.provenance !== 'measured' || p.units !== 'ms' || !object(p.environment) || !Array.isArray(p.points) || !p.points.length || !Number.isInteger(p.sample_count) || Number(p.sample_count) < 1 || typeof p.evidence_sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(p.evidence_sha256) || !object(p.model_identity) || !object(p.model_identity.sender) || !object(p.model_identity.receiver)) throw new Error('泛化 Profile 需要实测来源、单位、身份、校准点及 SHA。')
    if (identity && JSON.stringify(identity) !== JSON.stringify(p.model_identity)) {
      const canonical = (v: unknown): string => object(v) ? JSON.stringify(Object.keys(v).sort().map(k => [k, canonical(v[k])])) : JSON.stringify(v)
      if (canonical(identity) !== canonical(p.model_identity)) throw new Error('Profile 模型身份与当前后端不匹配。')
    }
    return p as unknown as CostProfile
  }
  if (!object(p) || Object.keys(p).some(k => !profileNames.includes(k))) throw new Error('Profile 字段不符合增量 schema；不接受服务器文件路径。')
  if (!['version', 'notes'].every(k => typeof p[k] === 'string' && (p[k] as string).trim())) throw new Error('Profile 需要非空 version 和 notes。')
  if (!['measured_incremental', 'synthetic/test_only'].includes(String(p.provenance))) throw new Error('provenance 必须为 measured_incremental 或 synthetic/test_only。')
  if (p.semantics !== undefined && p.semantics !== 'decode_incremental_handoff_v1') throw new Error('不支持 Canonical profile；需要 decode_incremental_handoff_v1。')
  if (!object(p.model_identity) || !object(p.model_identity.sender) || !object(p.model_identity.receiver) || !Object.keys(p.model_identity.sender).length || !Object.keys(p.model_identity.receiver).length) throw new Error('Profile 需要 sender/receiver 模型身份。')
  const canonical = (v: unknown): string => object(v) ? JSON.stringify(Object.keys(v).sort().map(k => [k, canonical(v[k])])) : JSON.stringify(v)
  if (identity && canonical(identity) !== canonical(p.model_identity)) throw new Error('Profile 模型身份与当前后端不匹配（模型、revision、配置、dtype 或设备）。')
  if (!Number.isInteger(p.sample_count) || Number(p.sample_count) < (p.provenance === 'measured_incremental' ? 1 : 0)) throw new Error('measured Profile 需要实际 sample_count ≥ 1；synthetic 可为 0。')
  for (const k of ['message_token_range', 'cache_token_range', 'recompute_token_range']) {
    const r = p[k]
    if (k === 'recompute_token_range' && r === undefined) continue
    if (!Array.isArray(r) || r.length !== 2 || r.some(x => !Number.isInteger(x) || x < 0) || r[0] > r[1]) throw new Error(`${k} 需要两个有序非负整数。`)
  }
  if (!object(p.stages) || Object.entries(p.stages).some(([k, v]) => !stageNames.includes(k) || !object(v) || !('fixed_ms' in v) || Object.entries(v).some(([n, x]) => !costNames.includes(n) || typeof x !== 'number' || !Number.isFinite(x) || x < 0))) throw new Error('stages 需要受支持的阶段与有限非负成本（含 fixed_ms）。')
  return p as unknown as CostProfile
}
export function networkAutoConfig(d: NetworkAutoDraft, mode: DiscussionConfig['mode'], identity?: Record<string, unknown> | null) {
  const bandwidth = Number(d.bandwidth), latency = Number(d.latency)
  if (d.networkMode !== 'shannon' && (!d.bandwidth.trim() || !Number.isFinite(bandwidth) || bandwidth <= 0)) throw new Error('带宽 Mbps 必须为有限正数。')
  if (!d.latency.trim() || !Number.isFinite(latency) || latency < 0) throw new Error('单向基础延迟 ms 必须为有限非负数。')
  const trace = d.networkMode === 'trace' ? d.trace.split(',').map(s => s.trim() ? Number(s) : NaN) : []
  if (trace.length > 1000 || trace.some(n => !Number.isFinite(n) || n <= 0)) throw new Error('轨迹需要 1–1000 个正数 Mbps，用逗号分隔。')
  const result: Partial<DiscussionConfig> = { network_mode: d.networkMode, bandwidth_mbps: bandwidth, base_latency_ms: latency, bandwidth_trace: trace }
  if (d.networkMode === 'shannon') { result.channel = validateChannel(d.channel); delete result.bandwidth_mbps }
  // This builder only creates new experiments; historical Session configs stay untouched.
  if (mode === 'auto') {
    const profile = d.idealized ? null : parseProfile(d.profileText, identity)
    const pair: ManualMode[] = ['token', `kv:${d.selectedKv}`]
    if (!profile && !d.idealized) throw new Error('性能实验需要显式输入或导入成本 Profile JSON。')
    if (d.selectedKv === 'v3' && !d.allowV3) throw new Error('KV V3 需要额外实验许可。')
    Object.assign(result, { auto_policy: 'performance_experiment', auto_kv_version: d.selectedKv, auto_candidates: pair, idealized_simulation: d.idealized, allow_experimental_v3: d.allowV3, cost_profile: profile, synthetic_assumptions: d.idealized ? parseSynthetic(d.synthetic) : null })
  }
  return result
}
export function draftFromConfig(c: DiscussionConfig): NetworkAutoDraft {
  return { synthetic: syntheticDraft(c.synthetic_assumptions ?? defaultSynthetic), channel: c.channel ?? { ...defaultChannel }, networkMode: c.network_mode ?? 'manual', bandwidth: String(c.bandwidth_mbps ?? 20), latency: String(c.base_latency_ms ?? 0), trace: c.bandwidth_trace?.length ? c.bandwidth_trace.join(',') : defaultNetworkAuto.trace, autoPolicy: c.auto_policy ?? 'quality_first', selectedKv: c.auto_kv_version ?? (c.auto_candidates?.find(x => x !== 'token')?.slice(3) as NetworkAutoDraft['selectedKv']) ?? 'v0', idealized: c.idealized_simulation ?? false, allowV3: c.allow_experimental_v3 ?? false, profileText: c.cost_profile ? JSON.stringify(c.cost_profile, null, 2) : '' }
}
