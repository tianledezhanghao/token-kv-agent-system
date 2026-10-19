import type { PromptConfig } from './types'

export function PromptView({ config }: { config: PromptConfig }) {
  const stageNames = {
    a_first: 'A 第一轮阶段指令',
    b_first:
      config.version === 'secret-sum-v1'
        ? 'B 第一轮阶段指令'
        : 'B 检查阶段指令',
    a_followup: 'A 后续轮次阶段指令',
    b_followup: 'B 后续轮次阶段指令',
    final: '最终总结阶段指令',
  }
  return (
    <div className="discussion-prompts">
      {(['a', 'b'] as const).map((agent) => (
        <div key={agent}>
          <strong>Agent {agent.toUpperCase()} 实际生效的 System Prompt</strong>
          <p className="discussion-note">
            来源：
            {config[`agent_${agent}`].source === 'default' ? '默认' : '自定义'}
          </p>
          <pre>{config[`agent_${agent}`].text}</pre>
        </div>
      ))}
      <details>
        <summary>阶段指令（只读）· {config.version}</summary>
        {Object.entries(config.stages).map(([key, text]) => (
          <div key={key}>
            <strong>{stageNames[key as keyof typeof stageNames]}</strong>
            <pre>{text}</pre>
          </div>
        ))}
      </details>
    </div>
  )
}
