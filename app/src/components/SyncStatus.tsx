import { api, type ScanStatus } from '../api'

interface Props {
  status: ScanStatus | null
  offline: boolean
  onChange: () => void
}

function describe(status: ScanStatus): { label: string; detail?: string; progress?: number } {
  const { done, total } = status.stages.thumbnail
  const progress = total > 0 ? done / total : undefined
  const counts = `${done.toLocaleString()} of ${total.toLocaleString()}`
  switch (status.state) {
    case 'discovering':
      return { label: 'Finding photos', detail: `${status.discovered.toLocaleString()} found` }
    case 'processing':
      return { label: 'Syncing', detail: counts, progress }
    case 'faces': {
      const faces = status.stages.faces
      return {
        label: 'Finding faces',
        detail: `${faces.done.toLocaleString()} of ${faces.total.toLocaleString()}`,
        progress: faces.total > 0 ? faces.done / faces.total : undefined,
      }
    }
    case 'scenes': {
      const scenes = status.stages.scenes
      return {
        label: 'Reading scenes',
        detail: `${scenes.done.toLocaleString()} of ${scenes.total.toLocaleString()}`,
        progress: scenes.total > 0 ? scenes.done / scenes.total : undefined,
      }
    }
    case 'paused':
      return { label: 'Paused', detail: total > done ? counts : undefined, progress }
    default: {
      const parts = [`${status.library.photos.toLocaleString()} photos`]
      if (status.library.duplicate_files > 0) {
        parts.push(`${status.library.duplicate_files.toLocaleString()} duplicate files hidden`)
      }
      return { label: 'Up to date', detail: parts.join(' · ') }
    }
  }
}

export function SyncStatus({ status, offline, onChange }: Props) {
  if (offline || !status) {
    return (
      <div className="sync" role="status">
        <span className={`dot ${offline ? 'dot-error' : ''}`} />
        <span className="sync-label">{offline ? 'Local service is not running' : 'Connecting'}</span>
      </div>
    )
  }

  const { label, detail, progress } = describe(status)
  const act = (call: () => Promise<unknown>) => () => {
    void call().then(onChange, onChange)
  }

  return (
    <div className="sync" role="status">
      <span className={`dot dot-${status.state}`} />
      <span className="sync-label">{label}</span>
      {detail && <span className="sync-detail">{detail}</span>}
      {progress !== undefined && status.state !== 'idle' && (
        <progress className="sync-bar" value={progress} max={1} />
      )}
      {status.state === 'idle' && (status.faces_note || status.scenes_note) && (
        <span className="sync-errors">
          {[status.faces_note, status.scenes_note].filter(Boolean).join(' · ')}
        </span>
      )}
      {status.errors > 0 && (
        <span className="sync-errors" title="Files that could not be read are skipped">
          {status.errors.toLocaleString()} unreadable
        </span>
      )}
      {status.state === 'paused' ? (
        <button onClick={act(api.resumeScan)}>Resume</button>
      ) : status.state === 'idle' ? (
        <button onClick={act(api.startScan)}>Rescan</button>
      ) : (
        <button onClick={act(api.pauseScan)}>Pause</button>
      )}
    </div>
  )
}
