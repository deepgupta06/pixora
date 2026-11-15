import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { api, isDesktop, type PhotoSummary, type PrivatePhoto, type PrivateView, type Timeline } from '../api'
import { Grid } from './Grid'

const BROWSER_FORMATS = /\.(jpe?g|png|webp)$/i

function formatSize(bytes: number): string {
  return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.ceil(bytes / 1024)} KB`
}

/**
 * Private folder browsing. The service keeps this folder's thumbnails in memory
 * only: nothing goes into the library, the thumbnail cache, or the webview's cache.
 */
export function PrivateBrowse({ onExit }: { onExit: () => void }) {
  const [view, setView] = useState<PrivateView | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [path, setPath] = useState('')
  const [busy, setBusy] = useState(false)
  const [openId, setOpenId] = useState<number | null>(null)
  const tokenRef = useRef<string | null>(null)

  const close = useCallback(() => {
    const token = tokenRef.current
    tokenRef.current = null
    if (token) void api.privateClose(token).catch(() => {})
  }, [])

  // End the session when leaving private mode or closing the window.
  useEffect(() => {
    window.addEventListener('pagehide', close)
    return () => {
      window.removeEventListener('pagehide', close)
      close()
    }
  }, [close])

  const open = async (value: string) => {
    if (!value.trim()) return
    setBusy(true)
    setError(null)
    try {
      close()
      const status = await api.privateOpen(value)
      tokenRef.current = status.token
      setView({ ...status, photos: [] })
      setOpenId(null)
      setPath('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not open that folder')
    } finally {
      setBusy(false)
    }
  }

  const browse = async () => {
    const { open: pick } = await import('@tauri-apps/plugin-dialog')
    const picked = await pick({ directory: true, title: 'Choose a folder to look through privately' })
    if (typeof picked === 'string') void open(picked)
  }

  // Poll while the folder is being read; stop once it is ready.
  const token = view?.token
  const ready = view?.state === 'ready'
  useEffect(() => {
    if (!token || ready) return
    let alive = true
    const tick = async () => {
      try {
        const next = await api.privateView(token)
        if (alive && tokenRef.current === token) setView(next)
      } catch (err) {
        if (alive) setError(err instanceof Error ? err.message : 'The private session ended')
      }
    }
    const id = setInterval(() => void tick(), 700)
    void tick()
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [token, ready])

  // The grid groups photos by month, all of them already loaded.
  const { timeline, months, byId, ordered } = useMemo(() => {
    const months: Record<string, PhotoSummary[]> = {}
    const byId = new Map<number, PrivatePhoto>()
    for (const p of view?.photos ?? []) {
      byId.set(p.id, p)
      ;(months[p.taken_at.slice(0, 7)] ??= []).push({
        id: p.id,
        hash: String(p.id),
        taken_at: p.taken_at,
        width: p.width,
        height: p.height,
      })
    }
    const buckets = Object.keys(months)
      .sort()
      .reverse()
      .map((month) => ({ month, count: months[month].length }))
    const timeline: Timeline = { total: byId.size, buckets }
    return { timeline, months, byId, ordered: buckets.flatMap((b) => months[b.month]) }
  }, [view?.photos])

  const thumbUrl = useCallback((p: PhotoSummary) => api.privateThumbUrl(token ?? '', p.id, 256), [token])
  const loadMonth = useCallback(() => {}, [])
  const onOpen = useCallback((p: PhotoSummary) => setOpenId(p.id), [])

  const openIndex = openId === null ? -1 : ordered.findIndex((p) => p.id === openId)
  const current = openIndex >= 0 ? byId.get(ordered[openIndex].id) : undefined

  const form = (
    <form
      className="add-folder"
      onSubmit={(e: FormEvent) => {
        e.preventDefault()
        void open(path)
      }}
    >
      <div className="add-folder-row">
        {isDesktop && (
          <button type="button" className="primary" disabled={busy} onClick={() => void browse()}>
            Choose folder…
          </button>
        )}
        <input
          value={path}
          onChange={(e) => setPath(e.target.value)}
          placeholder={isDesktop ? 'or paste a folder path' : 'C:\\Users\\you\\Pictures'}
          aria-label="Folder path"
          spellCheck={false}
          autoComplete="off"
        />
        <button className={isDesktop ? undefined : 'primary'} disabled={busy || !path.trim()}>
          Open privately
        </button>
      </div>
      {error && <p className="form-error">{error}</p>}
    </form>
  )

  const working = view && view.state !== 'ready'

  return (
    <div className="app private">
      <header className="topbar">
        <h1>Pixora</h1>
        <span className="private-badge">Private · nothing is saved</span>
        {view && (
          <span className="private-folder muted" title={view.root}>
            {view.root}
          </span>
        )}
        <span className="private-progress muted">
          {view &&
            (working
              ? view.state === 'discovering'
                ? 'Looking for photos…'
                : `Reading ${view.done.toLocaleString()} of ${view.total.toLocaleString()}`
              : `${timeline.total.toLocaleString()} photos`)}
          {view && view.errors > 0 && ` · ${view.errors} could not be read`}
        </span>
        {view && (
          <button onClick={() => (close(), setView(null), setError(null))}>Another folder</button>
        )}
        <button className="primary" onClick={onExit}>
          Exit private
        </button>
      </header>

      {!view ? (
        <main className="empty">
          <h2>Look through a folder privately</h2>
          <p className="muted">
            Pixora shows the photos without adding the folder to your library. No thumbnails, faces, or
            details are saved, and everything is forgotten when you exit.
          </p>
          {form}
        </main>
      ) : timeline.total === 0 ? (
        <main className="empty">
          <h2>{working ? 'Looking for photos…' : 'No photos in this folder'}</h2>
          {error && <p className="form-error">{error}</p>}
        </main>
      ) : (
        <>
          {view.truncated && (
            <p className="private-note muted">
              This folder has more photos than private browsing holds; only the first{' '}
              {view.total.toLocaleString()} are shown.
            </p>
          )}
          <Grid key={view.token} timeline={timeline} months={months} loadMonth={loadMonth} onOpen={onOpen} thumbUrl={thumbUrl} />
        </>
      )}

      {current && token && (
        <PrivateViewer
          token={token}
          photo={current}
          onClose={() => setOpenId(null)}
          onPrev={openIndex > 0 ? () => setOpenId(ordered[openIndex - 1].id) : undefined}
          onNext={openIndex < ordered.length - 1 ? () => setOpenId(ordered[openIndex + 1].id) : undefined}
        />
      )}
    </div>
  )
}

interface ViewerProps {
  token: string
  photo: PrivatePhoto
  onClose: () => void
  onPrev?: () => void
  onNext?: () => void
}

function PrivateViewer({ token, photo, onClose, onPrev, onNext }: ViewerProps) {
  // The large view shows at once; the original replaces it when loaded.
  const [fullFor, setFullFor] = useState<number | null>(null)

  useEffect(() => {
    if (!BROWSER_FORMATS.test(photo.name)) return
    let current = true
    const img = new Image()
    img.onload = () => current && setFullFor(photo.id)
    img.src = api.privateOriginalUrl(token, photo.id)
    return () => {
      current = false
    }
  }, [token, photo.id, photo.name])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      else if (e.key === 'ArrowLeft') onPrev?.()
      else if (e.key === 'ArrowRight') onNext?.()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, onPrev, onNext])

  const src =
    fullFor === photo.id ? api.privateOriginalUrl(token, photo.id) : api.privateThumbUrl(token, photo.id, 1024)
  const taken = new Date(photo.taken_at)

  return (
    <div className="viewer" role="dialog" aria-modal="true" aria-label="Photo viewer">
      <div className="viewer-stage" onClick={onClose}>
        <img key={photo.id} src={src} alt="" onClick={(e) => e.stopPropagation()} />
        {onPrev && (
          <button className="viewer-nav viewer-prev" aria-label="Previous photo" onClick={(e) => (e.stopPropagation(), onPrev())}>
            ‹
          </button>
        )}
        {onNext && (
          <button className="viewer-nav viewer-next" aria-label="Next photo" onClick={(e) => (e.stopPropagation(), onNext())}>
            ›
          </button>
        )}
      </div>
      <aside className="viewer-info">
        <header>
          <h2>Details</h2>
          <button onClick={onClose} aria-label="Close viewer">
            ✕
          </button>
        </header>
        <dl>
          <dt>Taken</dt>
          <dd>
            {taken.toLocaleDateString(undefined, { dateStyle: 'long' })}{' '}
            {taken.toLocaleTimeString(undefined, { timeStyle: 'short' })}
            {photo.taken_at_source === 'file' && <div className="muted">From the file date, no EXIF date</div>}
          </dd>
          <dt>Size</dt>
          <dd>
            {photo.width.toLocaleString()} × {photo.height.toLocaleString()}
          </dd>
          {photo.camera && (
            <>
              <dt>Camera</dt>
              <dd>{photo.camera}</dd>
            </>
          )}
          {photo.lat != null && photo.lon != null && (
            <>
              <dt>Location</dt>
              <dd>
                {photo.lat.toFixed(5)}, {photo.lon.toFixed(5)}
              </dd>
            </>
          )}
          <dt>File</dt>
          <dd>
            <div className="file-path">
              {photo.name}
              <span className="muted"> · {formatSize(photo.size)}</span>
            </div>
          </dd>
        </dl>
        <p className="muted private-note">Private: faces and scenes are not looked at, and nothing is kept.</p>
      </aside>
    </div>
  )
}
