import { useLayoutEffect, useState } from 'react'
import { EventLog, Modal, Properties, Tabs } from '../components/Workspace'
import { SessionConfig, type DiscussionController } from './DiscussionConfig'
import { DecisionFailure, DecisionInsights, NetworkHistory, ms } from './HandoffInsights'
import { modeLabel, phaseNames, receiptLabel, receiverCommitted } from './presentation'
import { AdvancedDiagnostic, DiagnosticFields, networkLabel, policyLabel, provenanceLabel, repairLabel, terminationLabel } from './DiagnosticFields'
import { LiveNetworkControl } from './LiveNetworkControl'
import type { Handoff } from './types'

export function Transmission({
  handoff: h,
  open = false,
}: {
  handoff: Handoff
  open?: boolean
}) {
  return (
    <details
      className={`trace-item ${h.error ? 'transfer-failed' : ''}`}
      open={open || undefined}
      id={`handoff-${h.sequence}`}
    >
      <summary>
        {h.source} → {h.target} ·{' '}
        {modeLabel(h.executed_mode, h.executed_kv_strategy)} ·{' '}
        {Number.isFinite(h.wire_bytes)
          ? `${h.wire_bytes.toLocaleString('en-US')} B`
          : '—'}{' '}
        ·{' '}
        {h.architecture === 'distributed' ? receiptLabel(h) : h.append_succeeded === true
          ? '已交付'
          : h.append_succeeded === false || h.error
            ? '失败'
            : '待确认'}
      </summary>
      <div className="trace-body">
        <DiagnosticFields title="通信决策" values={{
          '讨论轮次': h.discussion_round,
          '交接序号': h.sequence,
          '请求方式': modeLabel(h.requested_mode, h.requested_kv_strategy),
          '选中策略': modeLabel(h.selected_mode, h.selected_kv_strategy),
          '实际交付': receiverCommitted(h) && !h.error ? modeLabel(h.executed_mode, h.executed_kv_strategy) : '未完成交付',
          'AUTO 策略': h.auto_decision ? policyLabel(h.auto_decision.auto_policy) : '未使用 AUTO',
        }} />
        {h.auto_decision && <DecisionInsights decision={h.auto_decision} />}
        <DiagnosticFields title="网络环境" values={{ '模拟有效速率 Mbps': h.network?.bandwidth_mbps, 'B（MHz）':h.network?.channel?.bandwidth_mhz, 'SNR（dB）':h.network?.channel?.snr_db, '理论容量 C（Mbps）':h.network?.channel?.capacity_mbps, '假设 η':h.network?.channel?.efficiency, '单向基础延迟 ms': h.network?.base_latency_ms, '网络模式': networkLabel(h.network?.network_mode), '配置版本': h.network?.config_revision }} />
        {h.architecture === 'distributed' && <><DiagnosticFields title="真实 Relay / Receiver 提交" values={{
          '提交状态': receiptLabel(h), '来源 · measured': h.measurement_source ?? '未记录',
          '完整交接 Wall · observed': ms(h.observed_handoff_wall_ms),
          'Socket 应用交付 · measured': ms(h.socket_application_delivery_ms),
          'Receiver 处理 · measured': ms(h.receiver_processing_ms),
          '原始 protocol bytes': h.protocol_bytes, '应用封装 bytes': h.application_bytes,
          'Socket 实发 bytes': h.socket_measurement?.sent_bytes, 'Socket 实收 bytes': h.socket_measurement?.received_bytes,
          'Receiver 阶段': h.receiver_receipt?.stages, 'Receiver boot': h.receiver_receipt?.receiver_boot_id, 'packet SHA': h.receiver_receipt?.packet_sha256,
          'Cache 已导入并提交': h.receiver_receipt ? receiverCommitted(h) : undefined,
        }} /><p className="hint">三个真实计时窗口可能嵌套，不相加。Socket 包含连接、交付、Receiver 操作及应用 ACK。Mac loopback 不是无线或单向网络时延。Relay 转发量不计为新有效载荷。</p></>}
        <DiagnosticFields title="成本分析" values={{
          '预测本地处理': ms(h.predicted_local_ms, h.auto_decision?.profile_provenance === 'not_measured' ? '未校准' : '未预测'),
          '预测总计': ms(h.predicted_total_ms, h.auto_decision?.profile_provenance === 'not_measured' ? '未校准' : '未预测'),
          '模拟网络传输': ms(h.simulated_transfer_ms),
          '实测本地交接 · observed': ms(h.observed_local_ms, '不可用'),
          '预测交付字节': h.predicted_wire_bytes, '预测字节误差（actual − predicted）':h.wire_prediction_error_bytes,
          '实际交付字节': h.wire_bytes,
          '实测本地处理 + 模拟网络耗时': ms(h.observed_local_ms != null && h.simulated_transfer_ms != null ? h.observed_local_ms + h.simulated_transfer_ms : null),
        }} />
        <p className="hint">{h.architecture === 'distributed' ? '模拟网络与预测成本独立于上述 TCP 实测；不将模拟值加到 Socket 时间中。' : '本地处理为实测值，网络耗时为模拟值；组合值不是物理网络端到端耗时。'}</p>
        <DiagnosticFields title="Cache 与 KV 修正" values={{
          '修正方案': modeLabel(h.executed_mode, h.executed_kv_strategy),
          '实际修正操作': repairLabel(h.kv_repair_method),
          '接收方 Cache · 交付前': h.receiver_cache_length_before,
          '接收方 Cache · 交付后': h.receiver_cache_length_after,
          '发送方增量区间': h.sender_kv_span,
          '接收方目标区间': h.receiver_kv_target_span,
          '接收方消息预填充次数': h.receiver_message_prefill_count,
          '实际重算 Tokens': h.receiver_recomputed_token_count,
          '重算前向次数': h.recompute_forward_count,
          '修复 Token 索引': h.repaired_token_indices,
          '实际修复层': h.repaired_layer_indices,
          '层 × Token 工作量': h.repair_layer_token_count,
          '位置位移': h.rope_position_delta,
          '解包成功': h.unpack_succeeded,
          '追加成功': h.architecture === 'distributed' ? h.receiver_receipt ? receiverCommitted(h) : undefined : h.append_succeeded,
          'Cache 与账本一致': h.cache_ledger_consistent,
        }} />
        <DiagnosticFields title="交付内容" values={{ '消息内容 Tokens': h.message_content_token_count, '消息 ID': h.message_id, '错误': h.error, '回退说明': h.fallback_reason ?? '无回退' }} />
        <AdvancedDiagnostic value={h} title="高级诊断 · 完整交接元数据" />
      </div>
    </details>
  )
}
export default function DiscussionDetails({
  controller: d,
  onClose,
  initial = 'overview',
  selectedHandoff,
  debug,
  onDebugChange,
}: {
  controller: DiscussionController
  debug: boolean
  onDebugChange: (debug: boolean) => void
  onClose: () => void
  initial?: string
  selectedHandoff?: number
}) {
  const [tab, setTab] = useState(initial)
  const s = d.session,
    stats = s?.summary
  useLayoutEffect(() => {
    if (tab !== 'handoffs' || selectedHandoff === undefined) return
    const item = document.getElementById(`handoff-${selectedHandoff}`),
      body = item?.closest('.modal-body')
    if (item && body) {
      body.scrollTop +=
        item.getBoundingClientRect().top - body.getBoundingClientRect().top
      item.querySelector('summary')?.focus({ preventScroll: true })
    }
  }, [tab, selectedHandoff])
  return (
    <Modal
      title="实验详情"
      drawer
      onClose={onClose}
      description={
        s ? (
          <>
            Session {s.session_id}
            <br />
            {phaseNames[s.phase] ?? s.phase} · revision {s.revision}
          </>
        ) : (
          '尚未启动实验'
        )
      }
      tabs={
        <Tabs
          label="实验详情分区"
          value={tab}
          onChange={setTab}
          items={[
            { value: 'overview', label: '观测' },
            { value: 'handoffs', label: '通信证据' },
            { value: 'network', label: '网络轨迹' },
            { value: 'generation', label: '消息生成' },
            { value: 'logs', label: '日志' },
            { value: 'session', label: '本次配置' },
            { value: 'history', label: '会话' },
          ]}
        />
      }
    >
      <fieldset className="display-mode">
        <legend>页面显示</legend>
        <label><input type="radio" name="display-mode" checked={!debug} onChange={() => onDebugChange(false)} />展示</label>
        <label><input type="radio" name="display-mode" checked={debug} onChange={() => onDebugChange(true)} />调试</label>
      </fieldset>
      {d.architecture === 'distributed' && <><AdvancedDiagnostic value={d.latest?.health} title="Worker / Relay 健康状态与模型合约" />{s && s.status !== 'running' && !s.cache_released && <><p className="hint">仅保留现场诊断；不能恢复旧 Cache。显式释放不会撤销远端已经提交的数据。</p><button className="button" onClick={() => void d.release()}>释放分布式 Session</button></>}</>}
      {tab === 'overview' &&
        (s ? (
          <>
            <h3>实验结果</h3>
            <Properties
              values={{
                运行架构: s.architecture === 'distributed' ? '分布式双Agent' : '本地单进程',
                讨论状态: s.status === 'delivery_unknown' ? '提交结果未知 · 需对账' : s.status === 'completed' ? '已完成' : s.status === 'failed' ? '失败' : '进行中',
                完成原因: terminationLabel(s.termination_reason),
                失败阶段: s.failure_stage,
                错误: s.error,
                ...(stats
                  ? {
                      配置轮数: stats.rounds,
                      完成发言: stats.completed_turns,
                      成功通信次数: stats.handoff_count,
                      尝试通信次数: stats.attempted_handoff_count,
                      TOKEN: stats.token_handoffs,
                      'KV V0': stats.kv_v0_handoffs,
                      'KV V1': stats.kv_v1_handoffs,
                      'KV V2': stats.kv_v2_handoffs,
                      'KV V3': stats.kv_v3_handoffs,
                      '层 × Token 工作量': stats.repair_layer_token_count,
                      '实际重算 Tokens': stats.recomputed_tokens,
                      '总 wire bytes': `${stats.total_wire_bytes.toLocaleString('en-US')} B`,
                    }
                  : {
                      完成发言: s.turns.length,
                      已有交接记录: s.handoffs.length,
                      '已记录 wire bytes': `${s.handoffs.reduce((n, h) => n + (Number.isFinite(h.wire_bytes) ? h.wire_bytes : 0), 0).toLocaleString('en-US')} B`,
                    }),
              }}
            />
            <h3 className="section-space">独立 Agent Cache</h3>
            <Properties
              values={Object.fromEntries(
                Object.entries(s.cache_lengths).map(([a, n]) => [
                  `Agent ${a} ${s.cache_released ? '最终' : '当前'} Cache Tokens`,
                  n,
                ]),
              )}
            />
            <p className="hint">
              {s.cache_released
                ? 'Session Cache 已释放，保留最终计数'
                : 'Session Cache 由服务端持有'}
            </p>
            {s.score && (
              <>
                <h3 className="section-space">协作评分（事后核对）</h3>
                <Properties
                  unknownLabel="无法判定"
                  values={{
                    工程执行成功: s.score.engineering_success,
                    'A 发出自己的秘密': s.score.a_sent_own_secret,
                    'B 发出自己的秘密': s.score.b_sent_own_secret,
                    'B 正确使用 A 的秘密': s.score.b_received_a,
                    'A 正确使用 B 的秘密': s.score.a_received_b,
                    信息交换成功: s.score.information_exchange_success,
                    最终答案正确: s.score.final_sum_correct,
                    任务成功: s.score.task_success,
                    期望和: s.score.expected_sum,
                  }}
                />
                <p className="hint">
                  缺少或歧义评分显示“无法判定”。评分仅核对生成文本，单次猜中不证明依赖通信。
                  {s.score.note}
                </p>
              </>
            )}
            <h3 className="section-space">数据来源与边界</h3>
            <Properties
              values={{
                Transport: s.transport,
                模型: s.model?.model_name,
                Device: s.model?.device,
                Dtype: s.model?.dtype,
                创建时间: s.created_at,
                更新时间: s.updated_at,
              }}
            />
            <p className="hint">
              Wire bytes
              来自真实交接记录。网络耗时为模拟值，本地处理为实测值；没有物理网络 E2E 测量。工程执行成功与答案质量分别判断。
            </p>
            <details>
              <summary>完整 Session 诊断数据</summary>
              <pre className="raw-log">{JSON.stringify(s, null, 2)}</pre>
            </details>
          </>
        ) : (
          <p>尚无实验观测数据。</p>
        ))}
      {debug && tab === 'overview' && s && <DiagnosticFields title="调试环境" values={{ 'AUTO 策略': s.config.mode === 'auto' ? policyLabel(s.config.auto_policy) : '手动通信', '成本来源': provenanceLabel(s.config.idealized_simulation ? 'synthetic' : s.config.cost_profile?.provenance), '生成预算 Tokens': s.config.max_new_tokens }} />}
      {tab === 'generation' && (s ? <>
        <p className="hint">生成完成与通信交付分别记录；达到预算上限不表示正常结束。</p>
        {s.turns.map(turn => <section key={turn.message_id ?? turn.sequence}>
          <DiagnosticFields title={`Agent ${turn.agent} · 第 ${turn.round} 轮`} values={{ '生成 Tokens': turn.token_count, '配置预算 Tokens': turn.configured_max_new_tokens ?? s.config.max_new_tokens, '生效预算 Tokens': turn.effective_max_new_tokens, '可用上下文 Tokens': turn.context_remaining_tokens, '预算限制': turn.budget_limit_reason, '终止原因': terminationLabel(turn.termination_reason), '消息 ID': turn.message_id }} />
          <AdvancedDiagnostic value={turn} title="高级诊断 · 原始消息字段" />
        </section>)}
        {s.final_answer && <section><DiagnosticFields title={`最终总结 · Agent ${s.final_answer.agent}`} values={{ '生成 Tokens': s.final_answer.token_count, '配置预算 Tokens': s.final_answer.configured_max_new_tokens ?? s.config.max_new_tokens, '生效预算 Tokens': s.final_answer.effective_max_new_tokens, '可用上下文 Tokens': s.final_answer.context_remaining_tokens, '预算限制': s.final_answer.budget_limit_reason, '终止原因': terminationLabel(s.final_answer.termination_reason), '交付来源': '本地总结，无额外通信交付' }} /><AdvancedDiagnostic value={s.final_answer} title="高级诊断 · 原始总结字段" /></section>}
        {!s.turns.length && !s.final_answer && <p>尚无生成记录。</p>}
      </> : <p>尚无生成记录。</p>)}
      {tab === 'network'  && (s ? <><LiveNetworkControl controller={d} /><NetworkHistory session={s} /></> : <p>尚无网络快照。</p>)}
      {tab === 'handoffs' && <>
        {s?.active_handoff && <section id={`handoff-${s.active_handoff.sequence}`} className="trace-item"><h3>Handoff #{s.active_handoff.sequence} · 尚未交付</h3><DiagnosticFields title="通信决策" values={{ '请求方式': modeLabel(s.active_handoff.requested_mode ?? 'auto'), '选中策略': modeLabel(s.active_handoff.decision?.selected_candidate ?? s.active_handoff.auto_decision?.selected_candidate ?? s.active_handoff.selected_mode, s.active_handoff.selected_kv_strategy), '实际交付': '未完成交付' }} />{(s.active_handoff.decision ?? s.active_handoff.auto_decision) && <DecisionInsights decision={(s.active_handoff.decision ?? s.active_handoff.auto_decision)!} />}</section>}
        {s?.auto_decisions?.filter(d => d.decision_status === 'failed').map(d => <DecisionFailure key={d.handoff_index ?? d.sequence} record={d} />)}
        {s?.handoffs.length ? s.handoffs.map(h => <Transmission key={h.sequence} handoff={h} open={selectedHandoff === h.sequence} />) : <p>尚无交接证据。发言生成或决策选择不代表交付成功。</p>}
      </>}
      {tab === 'logs' && (
        <>
          {d.eventsError && (
            <p className="error-banner" role="alert">
              {d.eventsError}
              <button
                className="text-button"
                onClick={() => void d.refreshEvents()}
              >
                重试日志读取
              </button>
            </p>
          )}
          <EventLog events={d.events} labels={phaseNames} />
        </>
      )}
      {tab === 'history' && <section><h3>最近会话</h3><p className="hint">Local 保留进程内最近三次；Distributed 保存最近三次展示记录，重启不能恢复 Cache。打开历史不会重新启动讨论；当前会话链接支持刷新恢复。</p>{d.latest?.recent_sessions?.length ? d.latest.recent_sessions.map(record => <div className="history-entry" key={record.session_id}><strong>{record.question}</strong><p className="hint">{record.architecture === 'distributed' ? 'Distributed' : 'Local'} · {record.status} · {modeLabel(record.mode)} · {record.session_id}</p><button className="button" disabled={d.recovering || record.session_id === s?.session_id} onClick={() => void d.selectSession(record.session_id).then(() => setTab('handoffs'))}>{record.session_id === s?.session_id ? '当前会话' : '打开会话'}</button></div>) : <p>服务端未提供历史列表。</p>}</section>}
      {tab === 'session' && <SessionConfig controller={d} />}
    </Modal>
  )
}
