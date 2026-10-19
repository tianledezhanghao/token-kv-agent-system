import { SyntheticSummary } from './SyntheticCostFields'
import { useState } from 'react'
import { Modal, Tabs } from '../components/Workspace'
import { AutoSettings, NetworkSettings } from './NetworkAutoSettings'
import { LiveNetworkControl } from './LiveNetworkControl'
import { PromptView } from './PromptView'
import { provenanceLabel } from './DiagnosticFields'
import { manualModes, modeLabel } from './presentation'
import type { DiscussionConfig, ManualMode } from './types'
import type { useDiscussion } from './useDiscussion'

export type DiscussionController = ReturnType<typeof useDiscussion>
export function SessionConfig({
  controller: d,
}: {
  controller: DiscussionController
}) {
  const s = d.session
  return s ? (
    <>
      {s.config.idealized_simulation && <p className="profile-source">理想化仿真 · synthetic · 假设成本；本次 Session 固定</p>}
      {s.config.idealized_simulation && <SyntheticSummary value={s.config.synthetic_assumptions} />}
      {!s.config.idealized_simulation && s.config.cost_profile && <p className="profile-source">{provenanceLabel(s.config.cost_profile.provenance)}</p>}
      <pre className="raw-log">{JSON.stringify(s.config, null, 2)}</pre>
      <p className="hint">
        {s.model
          ? `${s.model.model_name} · ${s.model.device} · ${s.model.dtype}`
          : '模型信息未提供'}
      </p>
      <PromptView config={s.prompt_config} />
    </>
  ) : (
    <p>尚无已启动的实验。</p>
  )
}
export default function DiscussionConfigDialog({
  controller: d,
  onClose,
}: {
  controller: DiscussionController
  onClose: () => void
}) {
  const [tab, setTab] = useState('task')
  const [advancedRounds, setAdvancedRounds] = useState(d.rounds > 10)
  const { running, secret } = d
  if (d.architecture === 'distributed') return <Modal title="旧分布式会话配置" onClose={onClose}><p className="hint">仅查看历史；新实验请返回本地模式。</p><SessionConfig controller={d} /></Modal>
  if (running)
    return (
      <Modal
        title="实验配置"
        onClose={onClose}
        description="本次实验的冻结配置 · 模拟网络可独立调节"
      >
        {d.submitting && d.session?.status !== 'running' ? <p role="status">正在提交新实验，等待服务器确认。此前讨论记录保持不变。</p> : <><LiveNetworkControl controller={d} /><SessionConfig controller={d} /></>}
      </Modal>
    )
  return (
    <Modal
      title="实验配置"
      onClose={onClose}
      description="配置新实验草稿，当前讨论记录保持不变。"
      tabs={
        <Tabs
          label="配置分区"
          value={tab}
          onChange={setTab}
          items={[
            { value: 'task', label: '任务' },
            { value: 'agents', label: 'Agent' },
            { value: 'strategy', label: '通信' },
            { value: 'network', label: '网络' },
            { value: 'preview', label: '生效预览' },
          ]}
        />
      }
      footer={
        <>
          <span className="muted">
            {d.secret
              ? '一轮私有数字协作'
              : `${d.effectiveRounds} 轮 · ${modeLabel(d.mode)}`}
            {d.networkAuto.idealized && d.mode === 'auto' ? ' · 理想化仿真' : ''}
            {d.profileProvenance === 'synthetic/test_only' ? ' · Synthetic / Test Only' : ''}
          </span>
          <div className="modal-actions">
            <button className="button" onClick={onClose}>
              保留草稿
            </button>
            <button
              className="button primary"
              onClick={() =>
                void d.submit().then((ok) => {
                  if (ok) onClose()
                })
              }
              disabled={
                !d.valid ||
                !d.previewReady ||
                !d.latest?.available ||
                d.recovering
              }
            >
              {' '}
              {secret ? '开始协作' : '开始讨论'}
            </button>
          </div>
        </>
      }
    >
      <form onChange={d.markDraftEdited} onSubmit={(e) => e.preventDefault()}>
        {tab === 'task' && (
          <fieldset className="settings-fieldset" disabled={running}>
            <label className="field">
              <span>任务模式</span>
              <select
                aria-label="任务模式"
                value={d.scenario}
                onChange={(e) =>
                  d.setScenario(e.target.value as 'discussion' | 'secret_sum')
                }
              >
                <option value="discussion">普通讨论</option>
                <option value="secret_sum">秘密数字协作</option>
              </select>
            </label>
            {secret ? (
              <>
                <div className="fields-two">
                  <label className="field">
                    <span>Agent A 的秘密数字</span>
                    <input
                      aria-label="Agent A 的秘密数字"
                      type="number"
                      min={1}
                      max={99}
                      value={d.secretA}
                      onChange={(e) => d.setSecretA(e.target.value)}
                    />
                  </label>
                  <label className="field">
                    <span>Agent B 的秘密数字</span>
                    <input
                      aria-label="Agent B 的秘密数字"
                      type="number"
                      min={1}
                      max={99}
                      value={d.secretB}
                      onChange={(e) => d.setSecretB(e.target.value)}
                    />
                  </label>
                </div>
                <p className="hint">
                  每个模型初始只知道自己的数字。一次 A → B → A 协作，再由 A
                  总结。公共任务不含秘密数字。
                </p>
              </>
            ) : (
              <label className="field">
                <span>用户问题</span>
                <textarea
                  aria-label="用户问题"
                  maxLength={512}
                  rows={4}
                  value={d.question}
                  onChange={(e) => d.setQuestion(e.target.value)}
                />
              </label>
            )}
            <div className="fields-three">
              {!secret && (
                <label className="field">
                  <span>讨论轮数</span>
                  <select
                    aria-label="讨论轮数"
                    value={d.rounds}
                    onChange={(e) => d.setRounds(Number(e.target.value))}
                  >
                    {Array.from({length: advancedRounds ? 20 : 10},(_,i) => i+1).map((r) => (
                      <option value={r} key={r}>
                        {r}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <label className="field">
                <span>每次发言最大 Tokens</span>
                <input
                  aria-label="每次发言最大 Tokens"
                  type="number"
                  min={1}
                  list="generation-budgets"
                  value={d.tokens}
                  onChange={(e) => d.setTokens(e.target.value)}
                />
              </label>
              <div className="context-budget-field">
                <label className="field"><span>总上下文上限 Tokens</span><input aria-label="总上下文上限 Tokens" aria-describedby="context-budget-help" type="number" min="128" max="32768" value={d.contextTokens} onChange={e => d.setContextTokens(e.target.value)} /></label>
                <p id="context-budget-help" className="hint">每位 Agent 的上下文容量上限，包含提示词、历史消息及生成内容。</p>
              </div>
            <datalist id="generation-budgets">{[256, 512, 1024].map(n => <option key={n} value={n} />)}</datalist>
              <label className="field">
                <span>最终总结 Agent</span>
                <select
                  aria-label="最终总结 Agent"
                  disabled={secret}
                  value={secret ? 'A' : d.summarizer}
                  onChange={(e) => d.setSummarizer(e.target.value as 'A' | 'B')}
                >
                  <option value="A">Agent A</option>
                  <option value="B">Agent B</option>
                </select>
              </label>
            </div>
            {!secret && <label className="permission-check"><input type="checkbox" checked={advancedRounds} onChange={e => { setAdvancedRounds(e.target.checked); if (!e.target.checked && d.rounds > 10) d.setRounds(10) }} />高级轮数（最多20轮；先检查上下文与资源）</label>}
            <p className="hint">
              长讨论在创建前保守估算全部预算；超出上下文时拒绝，请减少轮数或生成预算。问题最多512字符。Token预算为正整数，建议256 / 512 / 1024；可输入小预算。可用上下文不足时，服务端明确报告生效预算。
            </p>
          </fieldset>
        )}
        {tab === 'agents' &&
          (secret ? (
            <p className="hint">
              秘密数字场景使用固定的私有 System
              Prompt，不接受自定义角色。普通讨论的角色草稿保留。
            </p>
          ) : (
            <fieldset className="settings-fieldset">
              <p className="hint">
                A/B 保持独立上下文。留空使用服务器默认
                Prompt；每个最多4000字符，仅对新实验生效。
              </p>
              {(['A', 'B'] as const).map((agent) => (
                <div key={agent}>
                  <label className="field">
                    <span>Agent {agent} System Prompt</span>
                    <textarea
                      aria-label={`Agent ${agent} System Prompt`}
                      maxLength={4000}
                      rows={5}
                      value={agent === 'A' ? d.agentA : d.agentB}
                      onChange={(e) =>
                        (agent === 'A' ? d.setAgentA : d.setAgentB)(
                          e.target.value,
                        )
                      }
                    />
                  </label>
                  <div className="field-actions">
                    <button
                      className="button quiet"
                      type="button"
                      onClick={() => {
                        d.markDraftEdited()
                        ;(agent === 'A' ? d.setAgentA : d.setAgentB)('')
                      }}
                    >
                      恢复 Agent {agent} 默认
                    </button>
                  </div>
                </div>
              ))}
              <button
                className="button"
                type="button"
                onClick={() => {
                  d.markDraftEdited()
                  d.setAgentA('')
                  d.setAgentB('')
                }}
              >
                恢复默认
              </button>
            </fieldset>
          ))}
        {tab === 'strategy' && (
          <fieldset className="settings-fieldset">
            <label className="field">
              <span>通信模式</span>
              <select
                aria-label="通信模式"
                value={d.mode}
                onChange={(e) =>
                  d.setMode(e.target.value as DiscussionConfig['mode'])
                }
              >
                {manualModes.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
                <option value="mixed">MIXED · 逐次 handoff</option>
                <option value="auto">
                  AUTO · 每次交接决策
                </option>
              </select>
            </label>
            {d.mode === 'auto' && <><AutoSettings controller={d} /></>}
            {d.mode === 'mixed' &&
              d.plan.slice(0, d.effectiveRounds * 2).map((mode, i) => (
                <label key={i} className="field">
                  <span>
                    Round {Math.floor(i / 2) + 1} · {i % 2 ? 'B → A' : 'A → B'}
                  </span>
                  <select
                    aria-label={`Handoff ${i + 1}`}
                    value={mode}
                    onChange={(e) =>
                      d.setPlan((p) =>
                        p.map((v, index) =>
                          index === i ? (e.target.value as ManualMode) : v,
                        ),
                      )
                    }
                  >
                    {manualModes.map((m) => (
                      <option key={m.value} value={m.value}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
            {d.hasV2 && <>
              <label className="field"><span>V2 重算策略</span>
                <select aria-label="V2 重算策略" value={d.recomputePolicy} onChange={e => { d.setRecomputePolicy(e.target.value as 'tokens' | 'ratio'); d.setRecomputeValue(e.target.value === 'ratio' ? '0.25' : '1') }}>
                  <option value="tokens">固定尾段 Token 数</option><option value="ratio">固定尾段比例</option>
                </select></label>
              <label className="field"><span>{d.recomputePolicy === 'tokens' ? 'V2 尾段 Token 数（超过消息长度时裁剪）' : 'V2 尾段比例（0–1，向上取整）'}</span>
                <input aria-label="V2 重算值" type="number" min="0" max={d.recomputePolicy === 'ratio' ? 1 : undefined} step={d.recomputePolicy === 'ratio' ? 0.01 : 1} value={d.recomputeValue} onChange={e => d.setRecomputeValue(e.target.value)} /></label>
            </>}
            <p className="hint">
              MIXED 是配置策略；每次实际交接分别使用 TOKEN 或具体 KV 版本。V0/V1/V2
              是实验性策略，不保证数值等价或加速。
            </p>
            <p className="hint">
              网络配置在“网络”分区。V2 仍传完整 KV；V3 已有质量退化且无数字时失败。配置仅对新 Session 生效。
            </p>
          </fieldset>
        )}
        {tab === 'network' && <NetworkSettings controller={d} />}
        {d.configurationError && <p className="error-text" role="alert">{d.configurationError}</p>}
        {tab === 'preview' &&
          (secret ? (
            <p className="hint">
              私有数字 Prompt
              由服务器在启动时生成，实际生效内容可在本次实验配置中查看。
            </p>
          ) : d.previewReady && d.preview ? (
            <PromptView config={d.preview.config} />
          ) : (
            <p className="hint">
              {d.promptError ?? '正在获取服务器实际 Prompt…'}
            </p>
          ))}
        {!d.valid && (
          <p className="error-text">
            请输入有效配置：问题非空，秘密数字1–99，Token预算为正整数；V2尾段数量为非负整数，或比例为0–1。
          </p>
        )}
        {d.promptError && (
          <p className="error-text" role="alert">
            Prompt 配置无效或无法预览：{d.promptError}
          </p>
        )}
        {d.error && (
          <p className="error-text" role="alert">
            {d.error}
          </p>
        )}
      </form>
    </Modal>
  )
}
