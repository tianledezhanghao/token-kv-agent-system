import { SyntheticCostFields } from './SyntheticCostFields'
import { ChannelFields } from './ChannelFields'
import { useState } from 'react'
import { parseProfile } from './autoConfig'
import { provenanceLabel } from './DiagnosticFields'
import { modeLabel } from './presentation'
import type { DiscussionController } from './DiscussionConfig'

export function AutoSettings({ controller: d }: { controller: DiscussionController }) {
  const n = d.networkAuto
  const [importError, setImportError] = useState<string | null>(null)
  let profile = null
  try { profile = parseProfile(n.profileText) } catch { /* Inline validation below. */ }
  return <>
    <label className="field"><span>AUTO 的 KV 版本</span><select aria-label="AUTO 的 KV 版本" value={n.selectedKv} onChange={e => d.setNetworkAuto(p => ({ ...p, selectedKv: e.target.value as typeof n.selectedKv }))}>{['v0','v1','v2','v3'].map(v => <option key={v} value={v}>KV {v.toUpperCase()}</option>)}</select></label>
    <p className="hint">当前候选：TOKEN ↔ {modeLabel(`kv:${n.selectedKv}`)} · {n.idealized ? '理想化场景仿真' : '性能实验'}</p>
    <p className="hint risk-note">性能实验允许指定的 KV 候选，已观察到语义风险，不保证可靠性或速度优势。</p>
    <label className="permission-check"><input type="checkbox" aria-label="理想化场景仿真" checked={n.idealized} onChange={e => d.setNetworkAuto(p => ({ ...p, idealized: e.target.checked }))} />理想化场景仿真</label>
    {n.idealized && <SyntheticCostFields value={n.synthetic} onChange={synthetic => { d.markDraftEdited(); d.setNetworkAuto(p => ({ ...p, synthetic })) }} />}
    {n.selectedKv === 'v3' && <label className="permission-check"><input type="checkbox" aria-label="额外许可 KV V3" checked={n.allowV3} onChange={e => d.setNetworkAuto(p => ({ ...p, allowV3: e.target.checked }))} />额外许可 KV V3 实验：已有质量退化，无合法数字槽位时明确排除。</label>}
    {!n.idealized && <><label className="field"><span>成本 Profile JSON（必填）</span><textarea aria-label="成本 Profile JSON" rows={7} value={n.profileText} onChange={e => d.setNetworkAuto(p => ({ ...p, profileText: e.target.value }))} placeholder="粘贴 generalized_incremental_cost_v1 或精确回放 Profile JSON" /></label>
    <label className="field"><span>导入本地 Profile JSON</span><input aria-label="导入本地 Profile JSON" type="file" accept=".json,application/json" onChange={async e => {
      const file = e.target.files?.[0]
      if (!file) return
      d.markDraftEdited()
      try {
        if (file.size > 1024 * 1024) throw new Error('Profile 超过 1 MiB，请使用精简的成本配置。')
        const text = await file.text()
        parseProfile(text, d.latest?.cost_model_identity)
        d.setNetworkAuto(p => ({ ...p, profileText: text }))
        setImportError(null)
      } catch (err) { setImportError(err instanceof Error ? err.message : '导入失败') }
      e.target.value = ''
    }} /></label>
    {importError && <p role="alert" className="error-text">{importError}</p>}
    {profile && <div className="profile-source"><strong>{profile.provenance === 'synthetic/test_only' ? 'Synthetic / Test Only · synthetic/test_only' : provenanceLabel(profile.provenance)}</strong><p className="hint">{profile.version} · {profile.sample_count} 样本 · 消息 {profile.message_token_range?.join('–') ?? '按校准点'} Tokens · Cache {profile.cache_token_range?.join('–') ?? '按校准点'} Tokens<br />{profile.notes}</p></div>}
    <p className="hint">文件在浏览器内解析为 JSON，仅提交配置。身份须匹配当前模型、revision、配置、dtype 与设备；不接受服务器路径。适用域和缺失阶段仍由真实 Selector 检查。</p>
    </>}
    {d.latest?.cost_model_identity && <details><summary>当前后端成本模型身份</summary><pre className="raw-log">{JSON.stringify(d.latest.cost_model_identity, null, 2)}</pre></details>}
  </>
}
export function NetworkSettings({ controller: d }: { controller: DiscussionController }) {
  const n = d.networkAuto
  return <fieldset className="settings-fieldset">
    <p className="hint">{d.architecture === 'distributed' ? '真实 Socket Relay；带宽与延迟为模拟参数，不节流实际 TCP。' : 'InMemoryByteTransport + 模拟网络耗时。不是物理限速。'}活动 Session 可在配置或网络详情中更新下一未冻结交接的网络。</p>
    <label className="field"><span>网络模式</span><select aria-label="网络模式" value={n.networkMode} onChange={e => d.setNetworkAuto(p => ({ ...p, networkMode: e.target.value as typeof n.networkMode }))}><option value="shannon">Shannon · 动态 SNR</option><option value="manual">历史模式 · 固定速率 Mbps</option><option value="trace">trace · 按 handoff 推进</option></select></label>
    {n.networkMode === 'shannon' && <ChannelFields value={n.channel} onChange={channel => d.setNetworkAuto(p => ({...p,channel}))} />}
    <div className="fields-two">{n.networkMode !== 'shannon' && <label className="field"><span>{n.networkMode === 'manual' ? '带宽 Mbps' : '初始配置带宽 Mbps（执行以轨迹为准）'}</span><input type="number" aria-label="带宽 Mbps" min="0" step="any" value={n.bandwidth} onChange={e => d.setNetworkAuto(p => ({ ...p, bandwidth: e.target.value }))} /></label>}
      <label className="field"><span>单向基础延迟 ms</span><input aria-label="单向基础延迟 ms" type="number" min="0" step="any" value={n.latency} onChange={e => d.setNetworkAuto(p => ({ ...p, latency: e.target.value }))} /></label></div>
    {n.networkMode === 'trace' && <label className="field"><span>带宽轨迹 Mbps（逗号分隔）</span><textarea aria-label="带宽轨迹 Mbps" rows={3} value={n.trace} onChange={e => d.setNetworkAuto(p => ({ ...p, trace: e.target.value }))} /></label>}
    {n.networkMode !== 'shannon' && <p className="hint">轨迹按 Handoff 1 / 2 / 3 的顺序推进，不是每秒改变一次。轨迹耗尽后 hold_last：保持末项。</p>}
  </fieldset>
}
