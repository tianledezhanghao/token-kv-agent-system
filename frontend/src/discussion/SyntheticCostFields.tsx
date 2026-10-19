import { coefficientNames, parseSynthetic, syntheticDraft, type SyntheticDraft } from './autoConfig'
import type { SyntheticAssumptions } from './types'

export function SyntheticSummary({ value }: { value?: SyntheticAssumptions | null }) {
  if (!value) return <p className="hint">旧记录未保存系数；使用当时的 idealized-reuse-v1 默认公式。</p>
  return <p className="hint">Synthetic · 假设成本（非实测 Profile）<br />TOKEN = {value.a} + {value.b}N + {value.c}NH ms；KV = {value.d} + {value.e}N + {value.f}k ms。</p>
}
export function SyntheticCostFields({ value = syntheticDraft(), onChange }: { value?: SyntheticDraft; onChange: (value: SyntheticDraft) => void }) {
  let parsed: SyntheticAssumptions | undefined, error: string | undefined
  try { parsed = parseSynthetic(value) } catch (e) { error = (e as Error).message }
  return <>
    <p className="hint risk-note">Synthetic · 假设计算成本，非实测 Profile。真实模型、KV 字节与协议照常执行，实际耗时独立测量。仅配置新实验；已创建 Session 的系数固定。</p>
    <details className="synthetic-settings"><summary>高级：计算成本假设</summary>
      <p className="hint">TOKEN = a + bN + cNH；KV = d + eN + fk。结果统一为毫秒；系数范围 0–1e9。N 为消息内容 Tokens（不含 EOS），H 为接收方缓存 Tokens。k 为当前算法修正的 Tokens：V0/V1 为 0，V2 为按数量或比例截取后的后缀长度，V3 为单个数字槽位（合法时为 1）。</p>
      <div className="fields-two">{coefficientNames.map(k => <label className="field" key={k}><span>{k} · {k < 'd' ? 'TOKEN' : 'KV'}</span><input aria-label={`假设系数 ${k}`} type="number" min="0" max="1000000000" step="any" value={value[k]} onChange={e => onChange({ ...value, [k]: e.target.value })} /></label>)}</div>
      <button type="button" className="button" onClick={() => onChange(syntheticDraft())}>恢复默认假设</button>
      {error ? <p role="alert" className="error-text">{error}</p> : <SyntheticSummary value={parsed} />}
    </details>
  </>
}
