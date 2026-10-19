import { useEffect, useRef, useState } from 'react'
import { ChannelFields } from './ChannelFields'
import { defaultChannel, validateChannel } from './autoConfig'
import type { DiscussionController } from './DiscussionConfig'

export function LiveNetworkControl({ controller: d }: { controller: DiscussionController }) {
  const s = d.session, network = s?.network
  const [channel, setChannel] = useState(defaultChannel)
  const [bandwidth, setBandwidth] = useState('20'), [latency, setLatency] = useState('0')
  const [override, setOverride] = useState(false), [busy, setBusy] = useState(false)
  const [feedback, setFeedback] = useState<string | null>(null), [error, setError] = useState<string | null>(null)
  const pending = useRef(false)
  useEffect(() => {
    setBandwidth(String(network?.config.bandwidth_mbps ?? s?.config.bandwidth_mbps ?? 20))
    setLatency(String(network?.config.base_latency_ms ?? s?.config.base_latency_ms ?? 0))
    setChannel(network?.config.channel ?? s?.config.channel ?? defaultChannel)
    setOverride(false)
  }, [s?.session_id, network?.config_revision, network?.config.bandwidth_mbps, network?.config.base_latency_ms, s?.config.bandwidth_mbps, s?.config.base_latency_ms])
  if (!s || s.status !== 'running') return null
  const noFutureHandoff = !!network && network.next_unfrozen_handoff_index > s.config.rounds * 2
  const trace = (network?.config.mode ?? s.config.network_mode) === 'trace'
  const shannon = network?.config.mode === 'shannon'
  let channelValid = true
  try { validateChannel(channel) } catch { channelValid = false }
  const valid = channelValid && !noFutureHandoff && bandwidth.trim() !== '' && latency.trim() !== '' && Number.isFinite(Number(bandwidth)) && Number(bandwidth) > 0 && Number.isFinite(Number(latency)) && Number(latency) >= 0 && (!trace || override)
  const apply = async () => {
    if (!valid || pending.current) return
    pending.current = true; setBusy(true); setFeedback(null); setError(null)
    try {
      const accepted = await d.updateNetwork(s.session_id, shannon ? { channel, base_latency_ms: Number(latency) } : { bandwidth_mbps: Number(bandwidth), base_latency_ms: Number(latency), manual_override: trace && override })
      setFeedback(`已更新模拟网络 · 版本 ${accepted.config_revision} · 从下一未冻结 Handoff #${accepted.effective_from_handoff_index} 生效${accepted.mode_before === 'trace' ? ' · 已切换到 manual override' : ''}`)
    } catch (e) { setError(e instanceof Error ? e.message : '网络更新失败；请恢复状态后重试。') }
    finally { pending.current = false; setBusy(false) }
  }
  return <section aria-label="活动模拟网络控制"><h3>实时模拟网络</h3>
    <p className="hint">已冻结交接保持原快照。生效顺序由服务器接受修改和冻结交接的先后决定；AUTO 在后续交接重新决策。</p>
    {noFutureHandoff && <p className="hint">本次全部交接已冻结，后续仅生成本地总结；网络修改已关闭。</p>}
    {shannon && <><ChannelFields value={channel} onChange={setChannel} disabled={busy} /><p className="hint">改变 B 会改变固定带宽实验条件，后续按配置版本分段记录。修改 seed 从新 seed 的首个 AR 状态开始，其余参数更新保留随机状态。</p></>}
    <div className="fields-two">{!shannon && <label className="field"><span>模拟带宽 Mbps</span><input aria-label="活动模拟带宽 Mbps" type="number" min="0" step="any" value={bandwidth} disabled={busy} onChange={e => setBandwidth(e.target.value)} /></label>}
      <label className="field"><span>单向基础延迟 ms</span><input aria-label="活动单向基础延迟 ms" type="number" min="0" step="any" value={latency} disabled={busy} onChange={e => setLatency(e.target.value)} /></label></div>
    {trace && <label className="permission-check"><input type="checkbox" checked={override} disabled={busy} onChange={e => setOverride(e.target.checked)} />切换到 manual override，停止推进 trace</label>}
    <button className="button" disabled={busy || !valid || d.recovering} aria-busy={busy} onClick={() => void apply()}>{busy ? '正在更新…' : '应用模拟网络'}</button>
    {feedback && <p className="hint" role="status">{feedback}</p>}{error && <p className="error-text" role="alert">{error}</p>}
  </section>
}
