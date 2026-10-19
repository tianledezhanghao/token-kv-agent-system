import { useId, useState } from 'react'
import { modeLabel, receiverCommitted } from './presentation'
import type { DiscussionSession, Handoff } from './types'

// Actual committed deliveries only: never infer execution from requested/selected modes.
export function timelineHandoffs(handoffs: Handoff[] = []) {
  return handoffs.filter(h => Number.isInteger(h.sequence) && h.sequence > 0 &&
    receiverCommitted(h) && !h.error && /^(token|kv(?::v\d+)?)$/i.test(h.executed_mode ?? ''))
    .sort((a, b) => a.sequence - b.sequence)
}

export function handoffDescription(h: Handoff) {
  const lines = [`交接 #${h.sequence} · ${h.source ? `Agent ${h.source}` : '发送方未记录'} → ${h.target ? `Agent ${h.target}` : '接收方未记录'}`, modeLabel(h.executed_mode, h.executed_kv_strategy)]
  const estimate = h.auto_decision?.candidate_estimates?.find(e => modeLabel(e.candidate) === modeLabel(h.executed_mode, h.executed_kv_strategy))
  const add = (label: string, value: number | null | undefined, unit: string) => {
    if (typeof value === 'number' && Number.isFinite(value)) lines.push(`${label}：${value.toFixed(2)} ${unit}`)
  }
  add('预测本地成本', h.predicted_local_ms ?? estimate?.predicted_local_ms, 'ms')
  add('预测模拟网络成本', estimate?.simulated_transfer_ms ?? h.simulated_transfer_ms, 'ms')
  add('预测总成本', h.predicted_total_ms ?? estimate?.predicted_total_ms, 'ms')
  add('实测本地交接', h.observed_local_ms, 'ms')
  const network = h.network ?? h.auto_decision?.network
  add('SNR', network?.channel?.snr_db, 'dB')
  add('理论容量 C', network?.channel?.capacity_mbps, 'Mbps')
  add('模拟有效速率 R', network?.channel?.effective_rate_mbps ?? network?.bandwidth_mbps, 'Mbps')
  return lines.join('\n')
}

export function ModeTimeline({ session, mini = false }: { session: DiscussionSession; mini?: boolean }) {
  const all = timelineHandoffs(session.handoffs)
  const rows = mini ? all.slice(-8) : all
  const id = useId()
  const [active, setActive] = useState<{ session: string; sequence: number } | null>(null)
  const selected = active?.session === session.session_id ? rows.find(h => h.sequence === active.sequence) : undefined
  // Full charts retain readable node spacing; only their own viewport scrolls.
  const width = mini ? 300 : Math.max(600, rows.length * 30 + 80)
  const height = mini ? 56 : 154
  const top = mini ? 12 : 32, bottom = mini ? 36 : 94
  const first = rows[0]?.sequence ?? 1, last = rows.at(-1)?.sequence ?? first
  // Categorical spacing keeps sparse historical sequences legible; labels retain real IDs.
  const x = (h: Handoff) => 48 + rows.indexOf(h) / Math.max(1, rows.length - 1) * (width - 70)
  const y = (h: Handoff) => h.executed_mode?.toLowerCase() === 'token' ? bottom : top
  const path = rows.map((h, i) => `${i ? 'H' : 'M'}${x(h)}${i ? `V${y(h)}` : ` ${y(h)}`}`).join(' ')
  const legend = '● A · ◆ B · 外环为最新'
  return <section className={`mode-timeline ${mini ? 'mode-timeline-mini' : 'mode-timeline-full'}`} aria-label={mini ? '最近通信模式切换' : '通信模式切换时间线'} onKeyDown={e => {
    if (e.key === 'Escape' && selected) { e.preventDefault(); e.stopPropagation(); setActive(null) }
  }}>
    {!mini && <><h3>通信模式切换</h3><p className="hint">已确认交付 {all.length} 次 · 圆点 = Agent A · 菱形 = Agent B · 外环 = 最新交接</p></>}
    {!rows.length ? <p className="timeline-empty">暂无已确认交接</p> : <>
      <div className="mode-timeline-scroll">
        <svg viewBox={`0 0 ${width} ${height}`} style={!mini && width > 600 ? { minWidth: width } : undefined} role="group" aria-label={`交接序号 #${first} 至 #${last}，上轨 KV，下轨 Token`}>
          <path className="timeline-rail" d={`M48 ${top}H${width - 22}M48 ${bottom}H${width - 22}`} />
          <text x="0" y={top + 4}>KV</text><text x="0" y={bottom + 4}>Token</text>
          <path className="timeline-step" d={path} />
          {rows.map(h => <g key={h.sequence} className={`timeline-node ${h.sequence === all.at(-1)?.sequence ? 'latest' : ''}`} transform={`translate(${x(h)} ${y(h)})`}
            tabIndex={0} role="button" aria-label={`${handoffDescription(h)}${h.sequence === all.at(-1)?.sequence ? '\n最新交接' : ''}`}
            aria-describedby={selected?.sequence === h.sequence ? id : undefined}
            onMouseEnter={() => setActive({ session: session.session_id, sequence: h.sequence })}
            onFocus={() => setActive({ session: session.session_id, sequence: h.sequence })}
            onClick={() => setActive({ session: session.session_id, sequence: h.sequence })}
            onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setActive({ session: session.session_id, sequence: h.sequence }) } }}>
            <rect className="timeline-hit" x="-12" y="-12" width="24" height="24" />
            {h.sequence === all.at(-1)?.sequence && <circle className="timeline-current" r={9} />}
            {h.source === 'A' ? <circle className="timeline-marker" r={4} /> : h.source === 'B' ? <path className="timeline-marker" d="M0 -5 5 0 0 5 -5 0Z" /> : <rect className="timeline-marker" x="-4" y="-4" width="8" height="8" />}
          </g>)}
          {rows.map((h, i) => (!mini || i === 0 || i === rows.length - 1) && <text key={h.sequence} className="timeline-index" x={x(h)} y={height - 5} textAnchor="middle">#{h.sequence}</text>)}
        </svg>
      </div>
      {mini ? <div className="timeline-caption"><span>交接序号</span><span>{legend}</span></div> : <p className="hint">交接序号（按交接推进，非真实秒数）· 悬停或聚焦节点查看详情{width > 600 ? ' · 可横向滚动' : ''}</p>}
      {selected && <div className="timeline-tooltip" role="tooltip" id={id}>
        {handoffDescription(selected)}{selected.sequence === all.at(-1)?.sequence ? '\n最新交接' : ''}
        <button type="button" aria-label="关闭交接提示" onClick={() => setActive(null)}>×</button>
      </div>}
    </>}
  </section>
}
