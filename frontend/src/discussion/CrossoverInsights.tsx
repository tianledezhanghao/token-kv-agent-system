import { useState } from 'react'
import type { AutoDecision } from './types'
import { modeLabel } from './presentation'

export function CrossoverInsights({ decision: d }: { decision: AutoDecision }) {
  const c = d.crossover
  const channel = d.network?.channel
  const [bandwidth,setBandwidth] = useState(String(channel?.bandwidth_mhz ?? 20))
  if (!c) return null
  const b = Number(bandwidth), rate = c.effective_rate_mbps
  const exponent = rate && b > 0 ? rate/((channel?.efficiency ?? .8)*b)*Math.LN2 : NaN
  const threshold = exponent > 50 ? 10/Math.LN10*exponent : 10*Math.log10(Math.expm1(exponent))
  return <section aria-label="临界条件预分析"><h3>临界条件预分析</h3>
    <p className="hint">{c.cost_source === 'synthetic_assumption' ? 'Synthetic · 理想化假设成本，不是性能验证。' : '仅使用本次消息条件下已通过合法性检查的 Profile 预测。'}不改变 AUTO 的真实决策。</p>
    {c.status === 'positive_crossover' ? <>
      <p>临界有效速率 R* = {rate!.toFixed(3)} Mbps；低于临界值偏向 {modeLabel(c.below)}，高于偏向 {modeLabel(c.above)}。</p>
      <label className="field"><span>预分析 B（MHz，仅分析）</span><input aria-label="预分析 B（MHz，仅分析）" type="number" min="0.001" max="100000" value={bandwidth} onChange={e => setBandwidth(e.target.value)} /></label>
      <p className="hint">{b >= .001 && b <= 100000 && Number.isFinite(threshold) ? `η=${channel?.efficiency ?? .8} 时 SNR*≈${threshold.toFixed(2)} dB。${channel ? Math.abs(threshold-channel.snr_mean_db)<=2*channel.snr_sigma_db ? '处于 μ±2σ 范围内；该范围不是保证边界。' : '当前 μ±2σ 未覆盖临界值，可探索 B 或消息条件。' : ''}` : '该输入下无可表示的有限 SNR 临界值。'}固定消息与历史条件后才可解释交叉点；Profile 误差和质量门槛仍可能保持 TOKEN。</p>
      {c.within_configured_clip_bounds === false && <p className="hint">当前 SNR 截断区间不覆盖临界值。</p>}
    </> : <p className="hint">{c.status === 'no_positive_crossover' ? c.preferred === 'equal' ? '本次条件两方案成本相同，不存在唯一交叉点。' : `本次条件不存在正的有限交叉点，偏向 ${modeLabel(c.preferred)}。` : '缺少两个合法候选的本地成本，不能给出临界速率。未校准的3B不能借用0.5B成本。'}</p>}
  </section>
}
