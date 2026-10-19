import { SyntheticSummary } from './SyntheticCostFields'
import { ModeTimeline } from './ModeTimeline'
import { ChannelHistory } from './ChannelHistory'
import { CrossoverInsights } from './CrossoverInsights'
import { AdvancedDiagnostic, DiagnosticFields, networkLabel, provenanceLabel } from './DiagnosticFields'
import { candidates } from './autoConfig'
import { modeLabel } from './presentation'
import type { AutoDecision, DecisionRecord, DiscussionSession } from './types'

export const exclusionReasons: Record<string, string> = {
  environment_profile_mismatch: 'Profile 硬件 / 适配器 / 实现版本不匹配',
  profile_condition_unverified: '终止 / pending / framing / 启动或修正条件未校准',
  profile_sparse_cell: '缺少联合插值所需的完整校准角点',
  profile_workload_domain_unverified: '旧 Profile 缺少精确工作负载域',
  token_reference_unavailable: '缺少 TOKEN 基准预测，不能宣称 KV 更优',
  v3_no_numeric_slot: 'V3 无合法数字槽位，未执行 KV',
  protocol_capacity: '超过 KV 协议容量',
  not_whitelisted: '不在白名单', quality_not_certified: '质量未认证', known_quality_regression: '已知质量退化',
  model_profile_mismatch: 'Profile 模型身份不匹配', profile_out_of_domain: 'Profile 超出消息 / Cache 适用范围',
  profile_recompute_out_of_domain: 'Profile 超出重算适用范围', missing_cost_profile: '缺少校准成本 / 阶段数据',
  missing_startup_cost_profile: '缺少首次 KV 启动成本', unsupported_empty_kv: '空 KV 内容不支持', unsupported_strategy: '当前策略不支持',
}
export const ms = (n: number | null | undefined, missing = '未测量') => typeof n === 'number' && Number.isFinite(n) ? `${n.toFixed(2)} ms` : missing
export function DecisionInsights({ decision: d }: { decision: AutoDecision }) {
  return <section className="decision-insights" aria-label="AUTO 决策详情">
    <p className="hint">当前候选：{(d.candidate_set ?? d.candidate_estimates?.map(e => e.candidate) ?? []).map(c => modeLabel(c)).join(' ↔ ')}</p>
    <h3>AUTO 选择 {modeLabel(d.selected_candidate)}</h3>
    <p className="hint">{d.selection_reason === 'sole_safe_TOKEN_uncalibrated' ? (d.auto_policy === 'quality_first' ? '选择 TOKEN：当前仅 TOKEN 获得默认质量许可，本地成本未校准。' : '选择 TOKEN：这是唯一可用候选，本地成本未校准。') : ['minimum_predicted_pair_total_prefer_token_on_tie', 'minimum_predicted_total_then_token_v0_v1_v2_v3'].includes(d.selection_reason) ? '在 TOKEN 与指定 KV 之间比较预计总成本；同成本优先 TOKEN。' : d.selection_reason === 'uncertainty_overlap_prefer_token' ? '预测不确定性区间重叠，保守选择 TOKEN。' : '选择依据见高级诊断'}</p>
    <p className="diagnostic-source">{provenanceLabel(d.profile_provenance)}{d.profile_provenance === 'synthetic/test_only' ? ' · 仅用于路由测试，预测不代表实测性能' : ''}</p>
    {d.idealized && <SyntheticSummary value={d.synthetic_assumptions} />}
    {d.idealized && <p className="hint risk-note">理想化仿真：假设成本不证明真实系统性能优势，实际通信仍执行真实协议。</p>}
    {d.quality_risk && <p className="hint risk-note">性能实验：KV 存在语义风险，选中不等于交付成功。</p>}
    <div className="candidate-table-scroll"><table className="candidate-table"><caption>本次候选成本（预测）</caption><thead><tr><th>策略</th><th>本地</th><th>模拟网络</th><th>总计</th><th>不确定性</th><th>来源</th><th>状态</th></tr></thead><tbody>{(d.candidate_set ?? candidates).map(c => {
      const e = d.candidate_estimates?.find(e => e.candidate === c), excluded = d.excluded_candidates?.find(e => e.candidate === c)
      return <tr key={c} className={c === d.selected_candidate ? 'candidate-selected' : ''}><th scope="row">{modeLabel(c)}</th><td>{e ? ms(e.predicted_local_ms, e.cost_status === 'uncalibrated' ? '未校准' : '未提供') : '—'}</td><td>{e ? ms(e.simulated_transfer_ms) : '—'}</td><td>{e ? ms(e.predicted_total_ms, e.cost_status === 'uncalibrated' ? '未校准' : '未提供') : '—'}</td><td>{e ? e.cost_source === 'synthetic' ? '不适用（假设）' : ms(e.uncertainty_ms, '未提供') : '—'}</td><td>{e?.cost_source === 'synthetic' ? 'synthetic · 假设' : e?.cost_source === 'predicted' ? 'predicted · 校准预测' : '未提供'}</td><td>{excluded ? `已排除 · ${exclusionReasons[excluded.reason] ?? '其他原因（见高级诊断）'}` : e ? `${c === d.selected_candidate ? '已选中' : '可选'}${e.cost_source === 'synthetic' ? ' · 假设成本' : e.cost_status === 'profile_out_of_domain' ? ' · Profile 超出适用范围' : e.cost_status === 'uncalibrated' ? ' · 未校准' : ''}` : '未提供'}</td></tr>
    })}</tbody></table></div>
    <p className="hint">预计总计是预测本地处理 + 模拟网络耗时，非物理网络 E2E。排除项无成本排名。</p>
    <CrossoverInsights decision={d} />
    <DiagnosticFields title="成本模型" values={{ 'Profile 版本': d.prediction_config_version, '预测方法': d.prediction_method === 'bounded_joint_multilinear' ? '有界联合插值' : d.prediction_method === 'calibrated_point' ? '实测校准点预测' : d.idealized ? '理想化公式假设' : '旧阶段模型 / 未提供', '可靠性': d.reliability === 'experimental' ? '实验性预测，非误差保证' : d.idealized ? '假设，非性能验证' : d.reliability, '校准时间': d.profile_calibrated_at, '有效训练样本': d.profile_sample_count, '校准点 SHA': d.profile_evidence_sha256, '校准说明': d.profile_notes }} /><AdvancedDiagnostic value={d} title="高级诊断 · 原始 AUTO 决策" />
  </section>
}
export function DecisionFailure({ record: d }: { record: DecisionRecord }) {
  // The backend returns exclusions in its existing exception text. This parses reasons only, never associates messages by text.
  let exclusions: { candidate: string; reason: string }[] = []
  try { const start = d.error?.indexOf('[{') ?? -1; if (start >= 0) exclusions = JSON.parse(d.error!.slice(start)) } catch { /* Keep original error. */ }
  return <section className="error-text"><h3>Handoff #{d.handoff_index ?? d.sequence} · AUTO 决策失败 · 未交付</h3><p>没有完成可用策略选择，未进行交付。</p><AdvancedDiagnostic value={d} title="高级诊断 · 决策失败记录" />{exclusions.map(e => <p key={e.candidate}>{modeLabel(e.candidate)}：{exclusionReasons[e.reason] ?? '其他原因（见高级诊断）'}</p>)}</section>
}
export function sessionNetwork(s: DiscussionSession) {
  const active = s.active_handoff
  return active?.network ?? active?.decision?.network ?? active?.auto_decision?.network ?? s.handoffs.at(-1)?.network ?? s.auto_decisions?.at(-1)?.network
}
export function NetworkHistory({ session: s }: { session: DiscussionSession }) {
  const n = sessionNetwork(s), trace = s.config.bandwidth_trace ?? []
  const step = n?.network_mode === 'trace' ? n.simulation_step : undefined
  const next = step === undefined ? null : trace[Math.min(step + 1, trace.length - 1)]
  return <section className="network-diagnostics"><h3>网络环境</h3><p className="hint">内存传输 + 模拟网络。按交接推进，横轴是 Handoff 序号，不是时间或实时网速。已冻结交接保持原快照；更新仅影响下一尚未冻结交接。</p>
    <ModeTimeline session={s} />
    <ChannelHistory session={s} />
    <DiagnosticFields title="当前网络快照" values={{ '网络模式': networkLabel(n?.network_mode ?? s.config.network_mode ?? 'manual'), '当前交接序号': n?.handoff_index, [n?.channel ? '模拟有效速率 R（Mbps）' : '模拟带宽 Mbps']: n?.bandwidth_mbps, '单向基础延迟 ms': n?.base_latency_ms, '当前轨迹项': n?.network_mode === 'trace' && step !== undefined && trace.length ? `${Math.min(step, trace.length - 1) + 1} / ${trace.length}` : null, '下一项带宽 Mbps': n?.network_mode === 'trace' ? next : null, '轨迹状态': n?.network_mode === 'shannon' ? '动态 SNR · AR(1)' : n?.network_mode === 'trace' ? n.trace_exhausted ? '已耗尽，保持末项' : '按交接推进' : '固定配置' }} />
    {!!trace.length && <><h3 className="section-space">初始配置轨迹</h3><ol className="network-trace" aria-label="配置轨迹">{trace.map((v, i) => <li key={i} className={step !== undefined && Math.min(step, trace.length - 1) === i ? 'current' : ''}>Handoff {i + 1}<strong>{v} Mbps</strong></li>)}</ol></>}
    {s.network && <DiagnosticFields title="已接受网络配置" values={{ '网络模式': networkLabel(s.network.config.mode), '模拟有效速率 Mbps（旧模式）': s.network.config.mode === 'shannon' ? null : s.network.config.bandwidth_mbps, '单向基础延迟 ms': s.network.config.base_latency_ms, '配置版本': s.network.config_revision, '下一未冻结交接': s.network.next_unfrozen_handoff_index }} />}
    {!!s.network_updates?.length && <><h3 className="section-space">实时更新记录</h3>{s.network_updates.map(u => <details key={u.config_revision}><summary>版本 {u.config_revision} · 从下一未冻结 Handoff #{u.effective_from_handoff_index}</summary><DiagnosticFields title="服务器接受的修改" values={{ '接受时间': u.accepted_at, '原模式': networkLabel(u.mode_before), '新模式': networkLabel(u.mode_after), '原带宽 Mbps': u.before.config.bandwidth_mbps, '新带宽 Mbps（旧模式）': u.after.config.mode === 'shannon' ? null : u.after.config.bandwidth_mbps, '固定B是否改变': u.after.config.channel && u.before.config.channel ? u.after.config.channel.bandwidth_mhz !== u.before.config.channel.bandwidth_mhz : null, '新B（MHz）': u.after.config.channel?.bandwidth_mhz, '新SNR基准（dB）':u.after.config.channel?.snr_mean_db, '单向基础延迟 ms': u.after.config.base_latency_ms, '来源': u.source }} /></details>)}</>}
    <h3 className="section-space">交接快照</h3>
    {s.handoffs.map(h => <details className="network-record" key={h.sequence}><summary>交接 {h.sequence} · {h.source} → {h.target}</summary><DiagnosticFields title="网络参数" values={{ [h.network?.channel ? '模拟有效速率 R（Mbps）' : '模拟带宽 Mbps']: h.network?.bandwidth_mbps, '单向基础延迟 ms': h.network?.base_latency_ms, '模式': networkLabel(h.network?.network_mode), '轨迹状态': h.network?.network_mode === 'shannon' ? '动态 SNR · AR(1)' : h.network?.trace_exhausted ? '保持末项' : h.network?.network_mode === 'trace' ? '按交接推进' : '固定配置', '交付状态': h.append_succeeded && !h.error ? '已交付' : '失败' }} /><AdvancedDiagnostic value={h.network} title="高级诊断 · 原始网络快照" /></details>)}
    <AdvancedDiagnostic value={n} title="高级诊断 · 当前网络来源" />
    {!s.handoffs.length && <p className="hint">尚无交付 receipt；当前决策快照不代表交付成功。</p>}
  </section>
}
