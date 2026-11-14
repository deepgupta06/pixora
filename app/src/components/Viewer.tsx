import { useEffect, useState } from 'react'
import { api, type PhotoDetail, type PhotoSummary } from '../api'
import { FaceList } from './FaceList'

interface Props {
  photo: PhotoSummary
  onClose: () => void
  onPrev?: () => void
  onNext?: () => void
}

const BROWSER_FORMATS = /\.(jpe?g|png|webp)$/i

function capitalise(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1)
}

function formatSize(bytes: number): string {
  return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.ceil(bytes / 1024)} KB`
}

export function Viewer({ photo, onClose, onPrev, onNext }: Props) {
  const [detail, setDetail] = useState<PhotoDetail | null>(null)
  // The 1024 px thumbnail shows at once; the original replaces it when loaded.
  const [fullFor, setFullFor] = useState<number | null>(null)

  // Bumped after a correction so the people shown are read again.
  const [revision, setRevision] = useState(0)

  useEffect(() => {
    let current = true
    api.photo(photo.id).then(
      (d) => {
        if (!current) return
        setDetail(d)
        const primary = d.files.find((f) => !f.missing)
        if (primary && BROWSER_FORMATS.test(primary.path)) {
          const img = new Image()
          img.onload = () => current && setFullFor(photo.id)
          img.src = api.originalUrl(photo.id)
        }
      },
      () => {},
    )
    return () => {
      current = false
    }
  }, [photo.id, revision])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Typing a name must not flip through photos.
      if (e.target instanceof HTMLInputElement) return
      if (e.key === 'Escape') onClose()
      else if (e.key === 'ArrowLeft') onPrev?.()
      else if (e.key === 'ArrowRight') onNext?.()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, onPrev, onNext])

  const info = detail?.id === photo.id ? detail : null
  const src = fullFor === photo.id ? api.originalUrl(photo.id) : api.thumbUrl(photo.hash, 1024)
  const taken = new Date(photo.taken_at)

  return (
    <div className="viewer" role="dialog" aria-modal="true" aria-label="Photo viewer">
      <div className="viewer-stage" onClick={onClose}>
        <img key={photo.id} src={src} alt="" onClick={(e) => e.stopPropagation()} />
        {onPrev && (
          <button
            className="viewer-nav viewer-prev"
            aria-label="Previous photo"
            onClick={(e) => (e.stopPropagation(), onPrev())}
          >
            ‹
          </button>
        )}
        {onNext && (
          <button
            className="viewer-nav viewer-next"
            aria-label="Next photo"
            onClick={(e) => (e.stopPropagation(), onNext())}
          >
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
            {info?.taken_at_source === 'file' && <div className="muted">From the file date, no EXIF date</div>}
          </dd>
          <dt>Size</dt>
          <dd>
            {photo.width.toLocaleString()} × {photo.height.toLocaleString()}
          </dd>
          {info?.event && info.event.photo_count > 1 && (
            <>
              <dt>Event</dt>
              <dd>
                {[info.event.type ?? '', info.event.title].filter(Boolean).map(capitalise).join(' · ') || 'Outing'}
                <div className="muted">{info.event.photo_count.toLocaleString()} photos taken together</div>
              </dd>
            </>
          )}
          {info && info.tags.length > 0 && (
            <>
              <dt>Scene</dt>
              <dd>{info.tags.map((t) => capitalise(t.name)).join(', ')}</dd>
            </>
          )}
          {info?.camera && (
            <>
              <dt>Camera</dt>
              <dd>{info.camera}</dd>
            </>
          )}
          {info?.lat != null && info.lon != null && (
            <>
              <dt>Location</dt>
              <dd>
                {info.lat.toFixed(5)}, {info.lon.toFixed(5)}
              </dd>
            </>
          )}
          {info && info.faces.length > 0 && (
            <>
              <dt>People</dt>
              <dd>
                <FaceList faces={info.faces} onChange={() => setRevision((r) => r + 1)} />
              </dd>
            </>
          )}
          {info && (
            <>
              <dt>{info.files.length > 1 ? `Files (${info.files.length} copies)` : 'File'}</dt>
              <dd>
                {info.files.map((f) => (
                  <div key={f.path} className={`file-path ${f.missing ? 'muted' : ''}`}>
                    {f.path}
                    <span className="muted"> · {f.missing ? 'missing' : formatSize(f.size)}</span>
                  </div>
                ))}
              </dd>
            </>
          )}
        </dl>
      </aside>
    </div>
  )
}
