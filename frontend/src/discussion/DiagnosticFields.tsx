import { Properties } from '../components/Workspace'

export function DiagnosticFields({ title, values }: { title: string; values: Record<string, unknown> }) {
  return <section className="diagnostic-section" aria-label={title}>
    <h3>{title}</h3>
    <Properties values={values} unknownLabel="未报告" />
  </section>
}

export const terminationLabel = (reason?: string | null) => reason === 'eos' ? '正常结束（EOS）' : reason === 'max_new_tokens' ? '达到生成上限，内容可能未完成' : reason === 'context_limit' ? '上下文预算限制，内容可能未完成' : reason === 'resource_abort' ? '资源保护终止' : reason === 'error' ? '运行失败' : reason === 'rounds_completed' || reason === 'rounds_complete' ? '完成配置轮次' : reason ? '其他原因（见高级诊断）' : '未报告'
export const policyLabel = (policy?: string) => policy === 'quality_first' ? '质量优先' : policy === 'performance_experiment' ? '性能实验' : policy ? '其他策略（见高级诊断）' : '未报告'
export const provenanceLabel = (source?: string) => source === 'synthetic' ? '理想化假设 · synthetic' : source === 'measured' ? '实测校准 → 有界预测 · predicted' : source === 'synthetic/test_only' ? '模拟成本 · 测试' : source === 'measured_incremental' ? '实测校准成本' : source === 'not_measured' ? '未校准' : '未报告'
export const networkLabel = (mode?: string) => mode === 'shannon' ? 'Shannon · 动态 SNR' : mode === 'trace' ? '按交接推进的轨迹' : mode === 'manual' ? '手动配置' : '未报告'
export function AdvancedDiagnostic({ value, title = '高级诊断 · 原始字段与 JSON' }: { value: unknown; title?: string }) {
  return <details className="advanced-diagnostic"><summary>{title}</summary><pre className="raw-log">{JSON.stringify(value, null, 2)}</pre></details>
}

export function repairLabel(method?: string | null) {
  if (!method) return '未报告'
  return ({ raw_delta: '直接增量拼接', rope_aligned_delta: 'RoPE 位置修正', post_rope_rephase: 'RoPE 位置修正', suffix_recompute_delta: '连续尾段重算', numeric_slot_delta: '数字槽位修复' } as Record<string, string>)[method] ?? '其他操作（见高级诊断）'
}
