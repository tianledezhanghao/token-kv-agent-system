import { useState } from 'react'
import DiscussionPanel from './discussion/DiscussionPanel'
import CanonicalDashboard from './canonical/CanonicalDashboard'

export default function App() {
  const [tab, setTab] = useState<'discussion' | 'canonical'>('discussion')
  const [openedCanonical, setOpenedCanonical] = useState(false)
  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <svg viewBox="0 0 28 28" aria-hidden="true">
            <circle cx="8" cy="10" r="4" />
            <circle cx="20" cy="18" r="4" />
            <path d="m11 12 6 4" />
          </svg>
          <span>Agent Link</span>
        </div>
        <nav className="workspace-nav" aria-label="Dashboard 页面">
          <button
            aria-pressed={tab === 'discussion'}
            onClick={() => setTab('discussion')}
          >
            双 Agent 讨论
          </button>
          <button
            aria-pressed={tab === 'canonical'}
            onClick={() => {
              setOpenedCanonical(true)
              setTab('canonical')
            }}
          >
            通信基准
          </button>
        </nav>
      </header>
      <div className="workspace-host" hidden={tab !== 'discussion'}>
        <DiscussionPanel />
      </div>
      {openedCanonical && (
        <div className="workspace-host" hidden={tab !== 'canonical'}>
          <CanonicalDashboard />
        </div>
      )}
    </div>
  )
}
