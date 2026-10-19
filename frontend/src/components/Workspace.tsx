import { useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import type { ConnectionStatus, DashboardEvent } from '../types/dashboard'

export function Icon({
  name,
}: {
  name: 'settings' | 'plus' | 'detail' | 'close' | 'arrow'
}) {
  const paths = {
    settings: (
      <>
        <path d="M4 6h16M4 12h16M4 18h16" />
        <circle cx="9" cy="6" r="2" />
        <circle cx="15" cy="12" r="2" />
        <circle cx="9" cy="18" r="2" />
      </>
    ),
    plus: <path d="M12 4v16M4 12h16" />,
    detail: (
      <>
        <rect x="4" y="3" width="16" height="18" rx="1" />
        <path d="M9 3v18M13 8h3M13 12h3" />
      </>
    ),
    close: <path d="m6 6 12 12M18 6 6 18" />,
    arrow: <path d="M4 12h16m-6-6 6 6-6 6" />,
  }
  return (
    <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
      {paths[name]}
    </svg>
  )
}

export function Connection({
  status,
  syncing = false,
  retry,
  model,
  modelAvailable,
  modelOptions, modelLoading, modelSwitchDisabled, onModelSwitch,
}: {
  model?: { model_name: string; device: string; dtype?: string | null } | null
  modelOptions?: string[]
  modelLoading?: boolean
  modelSwitchDisabled?: boolean
  onModelSwitch?: (size: string) => void
  modelAvailable?: boolean
  status: ConnectionStatus
  syncing?: boolean
  retry?: () => void
}) {
  const [menuOpen, setMenuOpen] = useState(false)
  const [modelOpen, setModelOpen] = useState(false)
  const modelTrigger = useRef<HTMLButtonElement>(null)
  const cached = model?.model_name.match(/models--([^/]+)--([^/]+)\/snapshots\/([^/]+)/)
  const modelId = cached ? `${cached[1]}/${cached[2]}` : model?.model_name
  const shortName = modelId?.split('/').filter(Boolean).pop()
  const dtype = model?.dtype?.replace('torch.', '')
  const precisions: Record<string, string> = { bfloat16: 'BF16', float16: 'FP16', float32: 'FP32' }
  const precision = dtype ? (precisions[dtype] ?? dtype.toUpperCase()) : undefined
  return (
    <>
      <div
        className={`workspace-connection connection-${status}`}
        aria-live="polite"
      >
        <span className="connection-dot" />
        {syncing
          ? '状态同步中'
          : {
              connected: '已连接',
              reconnecting: '正在重连',
              disconnected: '未连接',
            }[status]}
        {modelOptions ? (
          <div className="model-picker" onBlur={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setMenuOpen(false)
          }} onKeyDown={(e) => {
            if (e.key === 'Escape') { setMenuOpen(false); modelTrigger.current?.focus() }
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
              e.preventDefault()
              setMenuOpen(true)
              const buttons = Array.from(e.currentTarget.querySelectorAll<HTMLButtonElement>('.model-menu button:not(:disabled)'))
              const index = buttons.indexOf(document.activeElement as HTMLButtonElement)
              buttons[(index + (e.key === 'ArrowDown' ? 1 : buttons.length - 1)) % buttons.length]?.focus()
            }
          }}>
            <button ref={modelTrigger} className="model-info-button"
              aria-label={modelLoading ? '正在加载模型' : `切换模型：${shortName ?? '模型未加载'}`} aria-expanded={menuOpen}
              aria-haspopup="menu" disabled={modelLoading}
              onClick={() => setMenuOpen(!menuOpen)}>
              {modelLoading ? '正在加载模型' : shortName ?? '模型未加载'}
            </button>
            {menuOpen && !modelLoading && <div className="model-menu" role="menu" aria-label="选择本地模型">
              {modelOptions.map(size => {
                const current = modelAvailable !== false && modelId === `Qwen/Qwen2.5-${size}-Instruct`
                return <button key={size} role="menuitemradio" aria-checked={current}
                  disabled={modelSwitchDisabled || current}
                  onClick={() => { setMenuOpen(false); onModelSwitch?.(size); modelTrigger.current?.focus() }}>
                  {size}{current ? ' · 当前模型' : ''}
                </button>
              })}
              {modelSwitchDisabled && <span className="hint">讨论进行中或状态同步中，暂不可切换</span>}
              {model && <button role="menuitem" onClick={() => { setMenuOpen(false); setModelOpen(true) }}>模型详情</button>}
            </div>}
          </div>
        ) : model && modelAvailable !== false ? (
          <button ref={modelTrigger} className="model-info-button"
            aria-label={`查看模型详情：${shortName}`} aria-haspopup="dialog"
            onClick={() => setModelOpen(true)}>{shortName}</button>
        ) : modelAvailable === false ? <span>模型未加载</span> : null}
        {retry && (
          <button className="text-button" onClick={retry}>
            重新同步
          </button>
        )}
      </div>
      {modelOpen && model && modelAvailable !== false && (
        <Modal title="模型详情" onClose={() => setModelOpen(false)} returnLabel="返回页面" returnFocus={modelTrigger.current}>
          <p className="hint">
            {status === 'connected' && !syncing
              ? '当前后端报告的已加载模型。'
              : '连接尚未恢复，以下为最近获取的模型信息。'}
          </p>
          <Properties values={{
            '模型': modelId,
            '设备': model.device?.toUpperCase(),
            '精度': precision,
            ...(cached ? { '快照版本': cached[3] } : {}),
            '加载来源': model.model_name,
          }} />
        </Modal>
      )}
    </>
  )
}

export function Modal({
  title,
  children,
  footer,
  onClose,
  drawer = false,
  tabs,
  description,
  returnLabel = '返回讨论',
  returnFocus,
}: {
  title: string
  children: ReactNode
  footer?: ReactNode
  onClose: () => void
  drawer?: boolean
  tabs?: ReactNode
  description?: ReactNode
  returnLabel?: string
  returnFocus?: HTMLElement | null
}) {
  const ref = useRef<HTMLDialogElement>(null)
  const closeRef = useRef(onClose)
  closeRef.current = onClose
  useLayoutEffect(() => {
    const previous = returnFocus ?? (document.activeElement as HTMLElement | null)
    const dialog = ref.current!
    if (!dialog.open) dialog.showModal()
    return () => {
      dialog.close()
      if (previous?.isConnected) previous.focus({ preventScroll: true })
    }
  }, [])
  return (
    <dialog
      ref={ref}
      className={drawer ? 'detail-drawer' : 'config-dialog'}
      aria-label={title}
      aria-modal="true"
      onCancel={(e) => {
        e.preventDefault()
        closeRef.current()
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          const r = e.currentTarget.getBoundingClientRect()
          if (
            e.clientX < r.left ||
            e.clientX > r.right ||
            e.clientY < r.top ||
            e.clientY > r.bottom
          )
            closeRef.current()
        }
      }}
    >
      <div className="modal-head">
        <h2>{title}</h2>
        <button
          className="button quiet icon-only"
          aria-label={`关闭${title}`}
          onClick={onClose}
          autoFocus
        >
          <Icon name="close" />
        </button>
      </div>
      {description && <div className="modal-context">{description}</div>}
      {tabs}
      <div className="modal-body scroll">{children}</div>
      <div className="modal-foot">
        {footer ?? (
          <>
            <span className="muted">阅读位置保持不变</span>
            <button className="button" onClick={onClose}>
              {returnLabel}
            </button>
          </>
        )}
      </div>
    </dialog>
  )
}

export function Tabs<T extends string>({
  label,
  items,
  value,
  onChange,
}: {
  label: string
  items: readonly { value: T; label: string }[]
  value: T
  onChange: (value: T) => void
}) {
  return (
    <nav className="modal-tabs" aria-label={label}>
      {items.map((item) => (
        <button
          key={item.value}
          className={`modal-tab ${value === item.value ? 'selected' : ''}`}
          aria-pressed={value === item.value}
          onClick={() => onChange(item.value)}
        >
          {item.label}
        </button>
      ))}
    </nav>
  )
}

export function Properties({
  values,
  unknownLabel = '—',
}: {
  values: Record<string, unknown>
  unknownLabel?: string
}) {
  return (
    <dl className="property-list">
      {Object.entries(values).map(([key, value]) => (
        <div className="property" key={key}>
          <dt>{key}</dt>
          <dd>
            {value === null ||
            value === undefined ||
            (typeof value === 'number' && !Number.isFinite(value))
              ? unknownLabel
              : typeof value === 'object'
                ? JSON.stringify(value)
                : typeof value === 'boolean'
                  ? value
                    ? '是'
                    : '否'
                  : String(value)}
          </dd>
        </div>
      ))}
    </dl>
  )
}

export function CommunicationStage({
  agents,
  status,
  source,
  target,
  mode,
  repair,
  onEvidence,
  pending = false,
  timeline,
}: {
  agents: { name: string; state: string }[]
  status: string
  source?: string | null
  target?: string | null
  mode: string
  repair?: string | null
  onEvidence?: () => void
  pending?: boolean
  timeline?: ReactNode
}) {
  return (
    <section
      className={`communication ${pending ? 'is-active' : ''} ${timeline ? 'has-mode-timeline' : ''}`}
      aria-label="双 Agent 通信状态"
    >
      <div className="communication-inner">
        <Agent agent={agents[0]} />
        <div className="channel">
          <div
            className={`channel-status ${pending ? 'running' : ''}`}
            role="status"
          >
            <span className="status-dot" />
            {status}
          </div>
          <div className="channel-route">
            <svg
              viewBox="0 0 148 18"
              className={`wire-arrow ${source?.replace('Agent ', '') === 'B' ? 'reverse' : ''}`}
              aria-label={
                source && target ? `${source} → ${target}` : '等待通信'
              }
            >
              <path
                d={
                  source && target
                    ? 'M0 9h145m-7-6 7 6-7 6'
                    : 'M3 9h142m-7-6 7 6-7 6M10 3 3 9l7 6'
                }
              />
            </svg>
            {onEvidence ? (
              <button className="stage-mode" onClick={onEvidence} title={mode}>
                <span className="stage-mode-copy">{mode}</span>
              </button>
            ) : (
              <span className="channel-mode" title={mode}><span className="stage-mode-copy">{mode}</span></span>
            )}
          </div>
          {repair && <div className="channel-repair">修正：{repair}</div>}
          {timeline}
        </div>
        <Agent agent={agents[1]} />
      </div>
    </section>
  )
}
function Agent({ agent }: { agent: { name: string; state: string } }) {
  return (
    <div className="agent">
      <span className="agent-avatar" aria-hidden="true">
        {agent.name.replace('Agent ', '')}
      </span>
      <div>
        <div className="agent-name">{agent.name}</div>
        <div className="agent-state">{agent.state}</div>
      </div>
    </div>
  )
}

export function ReadingSurface({
  stage,
  tabs,
  selected,
  onSelect,
  onDetails,
  viewKey,
  contentKey,
  children,
  footer,
  toolbarExtra,
}: {
  stage: ReactNode
  tabs: { value: string; label: string }[]
  selected: string
  onSelect: (key: string) => void
  onDetails: () => void
  viewKey: string
  contentKey: string
  children: ReactNode
  footer?: ReactNode
  toolbarExtra?: ReactNode
}) {
  const reader = useRef<HTMLDivElement>(null)
  const positions = useRef<Record<string, number>>({})
  const lastContent = useRef(contentKey)
  const follow = useRef(true)
  const [unread, setUnread] = useState(false)
  useLayoutEffect(() => {
    const el = reader.current!
    lastContent.current = contentKey
    el.scrollTop = positions.current[viewKey] ?? 0
    follow.current = el.scrollHeight - el.clientHeight - el.scrollTop < 32
    setUnread(false)
  }, [viewKey])
  useLayoutEffect(() => {
    if (lastContent.current === contentKey) return
    lastContent.current = contentKey
    if (selected === 'answer') return // A newly completed answer opens at its beginning.
    const el = reader.current!
    if (follow.current) {
      el.scrollTop = el.scrollHeight
      setUnread(false)
    } else setUnread(true)
  }, [contentKey])
  const select = (key: string) => {
    positions.current[viewKey] = reader.current?.scrollTop ?? 0
    onSelect(key)
  }
  return (
    <section className="reading-surface">
      {stage}
      <div className="reader-toolbar">
        <nav className="reading-tabs" aria-label="实验内容">
          {tabs.map((tab) => (
            <button
              key={tab.value}
              className={`reading-tab ${selected === tab.value ? 'selected' : ''}`}
              aria-pressed={selected === tab.value}
              onClick={() => select(tab.value)}
            >
              {tab.label}
            </button>
          ))}
        </nav>
        <div className="toolbar-actions">
          {toolbarExtra}
          <button className="button quiet" onClick={onDetails}>
            <Icon name="detail" />
            实验详情
          </button>
        </div>
      </div>
      <div
        className="reader"
        ref={reader}
        tabIndex={0}
        role="region"
        aria-label="实验内容阅读区"
        onScroll={() => {
          const el = reader.current!
          follow.current = el.scrollHeight - el.clientHeight - el.scrollTop < 32
          positions.current[viewKey] = el.scrollTop
          const keys = Object.keys(positions.current)
          if (keys.length > 6) delete positions.current[keys[0]]
          if (follow.current) setUnread(false)
        }}
      >
        <div className="reading-column">{children}</div>
      </div>
      <div className="reader-foot">
        <span>{footer}</span>
        {unread ? (
          <button
            className="text-action"
            onClick={() => {
              reader.current!.scrollTop = reader.current!.scrollHeight
              follow.current = true
              setUnread(false)
            }}
          >
            查看最新内容
            <Icon name="arrow" />
          </button>
        ) : (
          <span>完整文本 · 独立阅读区</span>
        )}
      </div>
    </section>
  )
}

export function EventLog({
  events,
  summary,
  labels = {},
}: {
  events: DashboardEvent[]
  summary?: (event: DashboardEvent) => string
  labels?: Record<string, string>
}) {
  const [filter, setFilter] = useState('all')
  const types = Array.from(new Set(events.map((e) => e.type))).sort()
  const displayed =
    filter === 'all' ? events : events.filter((e) => e.type === filter)
  return (
    <>
      <label className="field">
        <span>事件筛选</span>
        <select value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="all">全部事件</option>
          {types.map((t) => (
            <option key={t} value={t}>
              {labels[t] ?? t}
            </option>
          ))}
        </select>
      </label>
      <p className="hint">
        最近 {events.length} 个事件。服务器仅保留有界窗口，筛选不会删除记录。
      </p>
      {displayed.length ? (
        displayed.map((e, i) => (
          <details
            className="event-entry"
            key={`${e.timestamp}-${e.type}-${e.metadata.revision ?? i}`}
          >
            <summary>
              <time>
                {new Date(e.timestamp).toLocaleTimeString([], {
                  hour12: false,
                })}
              </time>
              <strong>{labels[e.type] ?? e.type}</strong>
            </summary>
            {summary && <p>{summary(e)}</p>}
            <p>
              {e.source}
              {e.target ? ` → ${e.target}` : ''}
            </p>
            <pre className="raw-log" translate="no">
              {JSON.stringify(e, null, 2)}
            </pre>
          </details>
        ))
      ) : (
        <p className="empty-state">暂无对应事件。</p>
      )}
    </>
  )
}
