import { defaultChannel } from './autoConfig'
import type { ShannonChannel } from './types'

export function ChannelFields({ value = defaultChannel, onChange, disabled = false }: {
  value?: ShannonChannel; onChange: (value: ShannonChannel) => void; disabled?: boolean
}) {
  const field = (key: keyof ShannonChannel, label: string, min: number, max: number, step: string | number = 'any', optional = false) =>
    <label className="field" key={key}><span>{label}</span><input aria-label={label} type="number" min={min} max={max} step={step}
      value={value[key] ?? ''} disabled={disabled} onChange={e => onChange({ ...value, [key]: optional && e.target.value === '' ? null : Number(e.target.value) })} /></label>
  return <>
    <div className="fields-two">
      {field('bandwidth_mhz','信道带宽 B（MHz）',.001,100000)}
      {field('snr_mean_db','基准 SNR μ（dB）',-120,120)}
      {field('snr_sigma_db','SNR 标准差 σ（dB）',0,40)}
    </div>
    <p className="hint">B 是频谱宽度；σ 是标准差，不是固定上下界。每次交接采样并冻结一次 SNR，不按真实秒数推进。</p>
    <details><summary>高级信道参数</summary><div className="fields-two">
      {field('correlation','相关系数 ρ',0,.999999)}
      {field('seed','随机种子 seed',0,4294967295,1)}
      {field('efficiency','容量折减 η（模拟假设）',.000001,1)}
      {field('snr_min_db','SNR 截断下界（dB，可留空）',-120,120,'any',true)}
      {field('snr_max_db','SNR 截断上界（dB，可留空）',-120,120,'any',true)}
    </div><p className="hint">C = B log₂(1 + 10^(SNR/10))；R = η C。η 为假设，C 为理论容量，均非硬件实测。AR(1) 保留时间相关性；相同配置与 seed 可重放。</p></details>
  </>
}
