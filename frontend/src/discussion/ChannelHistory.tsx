import { useState } from 'react'
import type { DiscussionSession } from './types'
import { DiagnosticFields } from './DiagnosticFields'
import { modeLabel } from './presentation'

export function ChannelHistory({ session: s }: { session: DiscussionSession }) {
  const [metric,setMetric] = useState<'snr_db'|'capacity_mbps'|'effective_rate_mbps'>('snr_db')
  const rows = s.handoffs.filter(h => h.network?.channel)
  if (!rows.length) return null
  const values = rows.map(h => h.network!.channel![metric])
  const low = Math.min(...values)-1, high = Math.max(...values)+1
  const x = (i: number) => 48+i*540/Math.max(1,rows.length-1)
  const y = (v: number) => 118-(v-low)/(high-low)*94
  const current = rows.at(-1)!.network!, channel = current.channel!
  const title = metric === 'snr_db' ? 'SNR（dB）' : metric === 'capacity_mbps' ? '理论容量 C（Mbps）' : '模拟有效速率 R（Mbps）'
  return <section className="channel-history" aria-label="动态信道轨迹">
    <h3>动态信道轨迹</h3>
    <label className="field"><span>曲线指标</span><select aria-label="曲线指标" value={metric} onChange={e => setMetric(e.target.value as typeof metric)}>
      <option value="snr_db">SNR（dB）</option><option value="capacity_mbps">理论容量 C（Mbps）</option><option value="effective_rate_mbps">模拟有效速率 R（Mbps）</option></select></label>
    <svg viewBox="0 0 640 156" role="img" aria-label={`${title}，横轴为交接序号，共${rows.length}个已记录样本`}>
      <path d="M48 16V118H600" fill="none" stroke="var(--muted)" />
      <text x="2" y="28">{high.toFixed(1)}</text><text x="2" y="118">{low.toFixed(1)}</text>
      {[...new Set(rows.map(h => h.network!.config_revision))].map(revision => <polyline key={revision} fill="none" stroke="var(--green)" strokeWidth="2"
        points={rows.map((h,i) => h.network!.config_revision === revision ? `${x(i)},${y(h.network!.channel![metric])}` : '').filter(Boolean).join(' ')} />)}
      {rows.map((h,i) => <circle key={h.sequence} cx={x(i)} cy={y(values[i])} r={i === rows.length-1 ? 5 : 3} fill="var(--green)"><title>{`交接${h.sequence}，v${h.network!.config_revision}，${values[i].toFixed(3)}，${modeLabel(h.executed_mode,h.executed_kv_strategy)}`}</title></circle>)}
      <text x="48" y="144">#{rows[0].sequence}</text><text x="480" y="144">当前 #{rows.at(-1)!.sequence} · 交接序号</text>
    </svg>
    <p className="hint">每个点是实际使用并保存的快照。配置版本分段绘制；横轴不是秒数。σ 是标准差，截断后不再保证原分布。</p>
    <DiagnosticFields title="当前香农信道" values={{'B（MHz）':channel.bandwidth_mhz,'SNR（dB）':Number(channel.snr_db.toFixed(3)),'理论容量 C（Mbps）':Number(channel.capacity_mbps.toFixed(3)),'假设 η':channel.efficiency,'模拟有效速率 R（Mbps）':Number(channel.effective_rate_mbps.toFixed(3)),'配置版本':current.config_revision,'seed':channel.seed}} />
    <div className="candidate-table-scroll"><table className="candidate-table"><caption>已执行交接与冻结信道</caption><thead><tr><th>序号 / 版本</th><th>B MHz</th><th>SNR dB</th><th>C Mbps</th><th>R Mbps</th><th>实际方式</th></tr></thead><tbody>{rows.map(h => {
      const c = h.network!.channel!
      return <tr key={h.sequence}><th scope="row">#{h.sequence} / v{h.network!.config_revision}</th><td>{c.bandwidth_mhz}</td><td>{c.snr_db.toFixed(2)}</td><td>{c.capacity_mbps.toFixed(2)}</td><td>{c.effective_rate_mbps.toFixed(2)}</td><td>{h.append_succeeded ? modeLabel(h.executed_mode,h.executed_kv_strategy) : '未交付'}</td></tr>
    })}</tbody></table></div>
  </section>
}
