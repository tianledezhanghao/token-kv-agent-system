import { useState } from 'react'
import {
  CommunicationStage,
  Connection,
  Icon,
  ReadingSurface,
} from '../components/Workspace'
import DiscussionConfigDialog from './DiscussionConfig'
import DiscussionDetails from './DiscussionDetails'
import {
  discussionStage,
  modeLabel,
  deliveryForTurn,
} from './presentation'
import { ms, sessionNetwork } from './HandoffInsights'
import { ModeTimeline } from './ModeTimeline'
import { useDiscussion } from './useDiscussion'

export default function DiscussionPanel() {
  const d = useDiscussion(),
    s = d.session
  const [configOpen, setConfigOpen] = useState(false)
  const [details, setDetails] = useState<{
    tab: string
    sequence?: number
  } | null>(null)
  const [reading, setReading] = useState('discussion')
  const [debug, setDebug] = useState(false)
  const stage = discussionStage(s)
  const snapshot = s ? sessionNetwork(s) : undefined
  const channel = snapshot?.channel
  const start = () =>
    void d.submit().then((ok) => {
      if (ok) setReading('discussion')
    })
  const openEvidence = (sequence?: number) =>
    setDetails({ tab: 'handoffs', sequence })
  const warnings = [
    d.latest?.model_switch?.error,
    d.recoveryError ? `无法恢复服务端状态：${d.recoveryError}` : null,
    !configOpen ? d.error : null,
    d.latest && !d.latest.available && !d.modelLoading
      ? d.architecture === 'distributed' ? '分布式服务未就绪。运行 .venv/bin/python run_distributed_dashboard.py --output-dir benchmark-results/distributed-dashboard-new' : '讨论模型未加载。请使用 --discussion --device mps 启动后端。'
      : null,
    s?.error
      ? `${s.status === 'delivery_unknown' ? '提交结果未知' : '讨论失败'} · ${s.failure_stage ?? '阶段未提供'}：${s.error}`
      : null,
  ].filter(Boolean)
  return (
    <>
      <Connection
        status={d.connection}
        model={d.architecture === 'distributed' ? d.latest?.health?.workers.A?.model ?? null : d.latest?.model}
        modelAvailable={d.latest?.available}
        modelOptions={d.architecture === 'local' ? d.latest?.model_switch?.options : undefined}
        modelLoading={d.modelLoading}
        modelSwitchDisabled={d.running || d.recovering || d.connection !== 'connected'}
        onModelSwitch={(size) => void d.changeModel(size)}
        syncing={d.recovering}
        retry={d.recoveryError ? () => void d.refresh() : undefined}
      />
      <main className="workspace" aria-label="双 Agent 讨论工作区">
        <section className="experiment-heading">
          <div className="title-block">
            <h1 title={s?.config.question}>
              {s
                ? s.config.scenario === 'secret_sum'
                  ? '秘密数字协作'
                  : s.config.question
                : '发起一次双 Agent 讨论'}
            </h1>
            <p className="context">
              {s
                ? `${s.config.rounds}轮交流 · ${modeLabel(s.config.mode)}`
                : '独立上下文，一起推敲一个问题'}
              {s?.config.idealized_simulation && <span title="Synthetic 假设成本，非实测 Profile"> · 理想化仿真</span>}
              {!s?.config.idealized_simulation && s?.config.cost_profile?.provenance === 'synthetic/test_only' && (
                <span title="Synthetic / Test Only：预测成本不是实测性能"> · 模拟成本 · 测试</span>
              )}
              {!s?.config.idealized_simulation && ['measured', 'measured_incremental'].includes(s?.config.cost_profile?.provenance ?? '') && <span> · 实测 Profile</span>}
              {s?.status === 'running' && d.connection !== 'connected'
                ? ' · 连接恢复中，状态待同步'
                : ''}
            </p>
          </div>
          <div className="actions">
            <button className="button" onClick={() => setConfigOpen(true)}>
              <Icon name="settings" />
              配置
            </button>
            {s ? (
              <button
                className="button primary"
                disabled={d.running}
                onClick={() => { d.newExperiment(); setConfigOpen(true) }}
              >
                <Icon name="plus" />
                {d.modelLoading ? '正在加载模型…' : d.running ? '讨论运行中…' : '新实验'}
              </button>
            ) : (
              <button
                className="button primary"
                onClick={start}
                disabled={
                  !d.loaded ||
                  d.running ||
                  !d.valid ||
                  !d.previewReady ||
                  !d.latest?.available ||
                  d.recovering
                }
              >
                {d.modelLoading ? '正在加载模型…' : d.running ? '讨论运行中…' : d.secret ? '开始协作' : '开始讨论'}
              </button>
            )}
          </div>
        </section>
        <ReadingSurface
          stage={
            <>
            <CommunicationStage
              {...stage}
              repair={null}
              timeline={s ? <ModeTimeline session={s} mini /> : undefined}
              onEvidence={
                stage.handoff
                  ? () => openEvidence(stage.handoff!.sequence)
                  : s?.active_handoff
                    ? () => openEvidence(s.active_handoff!.sequence)
                    : undefined
              }
            />
            </>
          }
          tabs={[
            { value: 'discussion', label: '讨论' },
            { value: 'answer', label: '最终答案' },
          ]}
          selected={reading}
          onSelect={setReading}
          onDetails={() => setDetails({ tab: 'overview' })}
          viewKey={`${s?.session_id ?? 'draft'}:${reading}`}
          contentKey={`${s?.session_id}:${s?.turns.length}:${s?.final_answer?.message_id ?? ''}`}
          footer={
            d.loaded
              ? s
                ? `${channel ? `SNR ${channel.snr_db.toFixed(1)} dB · C ${channel.capacity_mbps.toFixed(1)} / R ${channel.effective_rate_mbps.toFixed(1)} Mbps · v${snapshot?.config_revision} · ` : ''}更新于 ${new Date(s.updated_at).toLocaleTimeString([], { hour12: false })}`
                : '配置只对新实验生效'
              : '正在加载实验状态…'
          }
        >
          {d.architecture === 'distributed' && <p className="hint" role="status">旧分布式会话只读查看。<button className="button" onClick={() => d.changeArchitecture('local')}>返回本地实验</button></p>}
          {d.notice && <p className="hint" role="status">{d.notice}</p>}
          {warnings.length > 0 && (
            <div className="inline-notices">
              {warnings.map((w, i) => (
                <p className="error-banner" role="alert" key={i}>
                  {w}
                </p>
              ))}
            </div>
          )}
          {!s ? (
            <div className="preflight">
              <h2>从一个问题开始</h2>
              {!d.secret ? (
                <label className="field">
                  <span>实验问题</span>
                  <textarea
                    aria-label="实验问题"
                    maxLength={512}
                    rows={4}
                    placeholder="输入希望 Agent A 和 B 共同讨论的问题"
                    value={d.question}
                    onChange={(e) => {
                      d.markDraftEdited()
                      d.setQuestion(e.target.value)
                    }}
                  />
                </label>
              ) : (
                <p>秘密数字场景已选定，在配置中设置双方数字与通信策略。</p>
              )}
              <p className="preflight-description">
                {d.effectiveRounds} 轮讨论 · {modeLabel(d.mode)} · 每次发言最多{' '}
                {d.tokens} Tokens
                <br />
                完整角色、逐次交接计划与总结方式在配置中设置。
              </p>
              {d.promptError && !configOpen && (
                <p className="error-text" role="alert">
                  Prompt 预览失败：{d.promptError}
                </p>
              )}
              <button
                className="button quiet"
                onClick={() => setConfigOpen(true)}
              >
                调整实验配置
              </button>
            </div>
          ) : reading === 'discussion' ? (
            <>
              {[...s.turns]
                .sort((a, b) => a.sequence - b.sequence)
                .map((turn) => {
                  const reception = deliveryForTurn(s, turn)
                  return (
                    <article
                      className="turn"
                      key={`${s.session_id}:${turn.sequence}`}
                    >
                      <div className="speaker-mark" aria-hidden="true">
                        {turn.agent}
                      </div>
                      <div>
                        <div className="turn-head">
                          <strong>Agent {turn.agent}</strong>
                          <span className="round">
                            第
                            {['零', '一', '二', '三'][turn.round] ?? turn.round}
                            轮
                          </span>
                          {reception.handoff ? (
                            <button
                              className={`reception-label ${reception.failed ? 'failed' : ''}`}
                              onClick={() =>
                                openEvidence(reception.handoff!.sequence)
                              }
                              title="查看本条发言的交付证据"
                            >
                              {reception.label}{debug && ` · ${s.architecture === 'distributed' ? '完整交接' : '本地交接'} ${ms(s.architecture === 'distributed' ? reception.handoff?.observed_handoff_wall_ms : reception.handoff?.observed_local_ms, '不可用')}`}
                            </button>
                          ) : (
                            <span className="reception-label passive">
                              {reception.label}
                            </span>
                          )}
                        </div>
                        {['max_new_tokens', 'context_limit'].includes(turn.termination_reason) && <p className="hint" role="status">{turn.termination_reason === 'context_limit' ? '回答达到上下文预算限制' : '回答达到生成上限'}，内容可能未完成。</p>}
                        <div className="turn-text" translate="no">
                          {turn.text || '（模型生成内容为空）'}
                        </div>
                      </div>
                    </article>
                  )
                })}
              {!s.turns.length && (
                <p className="empty-state">
                  等待第一个完整发言。模型按发言完成后更新，不进行逐 Token
                  展示。
                </p>
              )}
              {s.status === 'running' && (
                <p className="awaiting">
                  {stage.status}
                  {s.agent ? ` · Agent ${s.agent}` : ''}
                </p>
              )}
            </>
          ) : (
            <div className="final-answer">
              {s.final_answer ? (
                <>
                  <p className="answer-credit">
                    最终总结 · Agent {s.final_answer.agent}
                  </p>
                  <h2>最终答案</h2>
                  {['max_new_tokens', 'context_limit'].includes(s.final_answer.termination_reason) && <p className="hint" role="status">{s.final_answer.termination_reason === 'context_limit' ? '回答达到上下文预算限制' : '回答达到生成上限'}，内容可能未完成。</p>}
                  <div className="answer-copy" translate="no">
                    {s.final_answer.text || '（模型生成内容为空）'}
                  </div>
                </>
              ) : (
                <>
                  <h2>尚无最终模型结果</h2>
                  <p className="preflight-description">
                    {s.status === 'running'
                      ? '讨论结束后，指定 Agent 将生成最终总结。'
                      : s.status === 'failed'
                        ? '实验未完成，已有发言和失败证据仍可查阅。'
                        : '服务器尚未提供最终答案。'}
                  </p>
                </>
              )}
            </div>
          )}
        </ReadingSurface>
      </main>
      {configOpen && (
        <DiscussionConfigDialog
          controller={d}
          onClose={() => setConfigOpen(false)}
        />
      )}{' '}
      {details && (
        <DiscussionDetails
          controller={d}
          debug={debug}
          onDebugChange={setDebug}
          initial={details.tab}
          selectedHandoff={details.sequence}
          onClose={() => setDetails(null)}
        />
      )}
    </>
  )
}
