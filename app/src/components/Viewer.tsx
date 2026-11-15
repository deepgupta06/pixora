import { useEffect, useRef, useState } from 'react'
import { api, type Creation, type PhotoDetail, type PhotoSummary } from '../api'
import { BrushCanvas, EditPanel, type BrushHandle, type EditTool } from './EditPanel'
import { FaceList } from './FaceList'
import { Compare, OriginBadge } from './StudioParts'

interface Props {
  photo: PhotoSummary
  onClose: () => void
  onPrev?: () => void
  onNext?: () => void
  /** Show another photo, such as an edit's result or the photo it came from. */
  onOpen: (photo: PhotoSummary) => void
  /** The library gained pictures. */
  onChange: () => void
  /** This picture, one Pixora made, was deleted. */
  onDeleted: () => void
  /** Start a new picture from the settings of a generated one. */
  onReuse: (creation: Creation) => void
}

const OPERATIONS: Record<string, string> = {
  remove_background: 'Background removed',
  replace_background: 'Background changed',
  enhance: 'Quality increased',
  restore: 'Restored',
  stylize: 'Restyled',
  inpaint: 'Retouched',
  extend: 'Extended',
  generate: 'Generated',
}

/** A result can be laid over its source only when both have the same shape. */
function sameShape(a: PhotoSummary, b: PhotoSummary): boolean {
  return Math.abs(a.width / a.height - b.width / b.height) < 0.01
}

const BROWSER_FORMATS = /\.(jpe?g|png|webp)$/i

function capitalise(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1)
}

function formatSize(bytes: number): string {
  return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.ceil(bytes / 1024)} KB`
}

export function Viewer({ photo, onClose, onPrev, onNext, onOpen, onChange, onDeleted, onReuse }: Props) {
  // The picture the user is being asked to confirm deleting, by photo id.
  const [deleteFor, setDeleteFor] = useState<number | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const confirmDelete = deleteFor === photo.id
  const remove = () => {
    setDeleting(true)
    setDeleteError(null)
    api.deleteCreation(photo.id).then(
      () => {
        setDeleting(false)
        setDeleteFor(null)
        onDeleted()
      },
      (e: unknown) => {
        setDeleting(false)
        setDeleteError(e instanceof Error ? e.message : 'Could not delete this picture')
      },
    )
  }
  const [editing, setEditing] = useState(false)
  const [tool, setTool] = useState<EditTool>('background')
  const [brushSize, setBrushSize] = useState(40)
  // The result being compared with its original, by photo id.
  const [compareFor, setCompareFor] = useState<number | null>(null)
  const comparing = compareFor === photo.id
  const image = useRef<HTMLImageElement>(null)
  const brush = useRef<BrushHandle>(null)
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
      const target = e.target
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement
      )
        return
      if (e.key === 'Escape') onClose()
      // While editing, the arrows must not leave the photo being worked on.
      else if (editing) return
      else if (e.key === 'ArrowLeft') onPrev?.()
      else if (e.key === 'ArrowRight') onNext?.()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, onPrev, onNext, editing])

  const info = detail?.id === photo.id ? detail : null
  const src = fullFor === photo.id ? api.originalUrl(photo.id) : api.thumbUrl(photo.hash, 1024)
  const taken = new Date(photo.taken_at)
  const creation = info?.creation ?? null
  const source = creation?.source && sameShape(creation.source, photo) ? creation.source : null
  const origin = info?.origin ?? photo.origin
  const painting = editing && tool === 'brush'

  return (
    <div className="viewer" role="dialog" aria-modal="true" aria-label="Photo viewer">
      <div className="viewer-stage" onClick={onClose}>
        {comparing && source && !editing ? (
          <Compare before={api.thumbUrl(source.hash, 1024)} after={src} />
        ) : (
          <img ref={image} key={photo.id} src={src} alt="" onClick={(e) => e.stopPropagation()} />
        )}
        {painting && <BrushCanvas key={photo.id} ref={brush} image={image} size={brushSize} />}
        {source && !editing && (
          <button
            className="viewer-compare"
            aria-pressed={comparing}
            onClick={(e) => (e.stopPropagation(), setCompareFor(comparing ? null : photo.id))}
          >
            {comparing ? 'Hide comparison' : 'Compare with original'}
          </button>
        )}
        {!editing && onPrev && (
          <button
            className="viewer-nav viewer-prev"
            aria-label="Previous photo"
            onClick={(e) => (e.stopPropagation(), onPrev())}
          >
            ‹
          </button>
        )}
        {!editing && onNext && (
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
          <h2>{editing ? 'Edit' : 'Details'}</h2>
          <div className="row">
            <button aria-pressed={editing} onClick={() => setEditing((e) => !e)}>
              {editing ? 'Done' : 'Edit'}
            </button>
            <button onClick={onClose} aria-label="Close viewer">
              ✕
            </button>
          </div>
        </header>
        {editing ? (
          <EditPanel
            photo={photo}
            tool={tool}
            onTool={setTool}
            brush={brush}
            brushSize={brushSize}
            onBrushSize={setBrushSize}
            onChange={onChange}
            onResult={(result) => {
              setEditing(false)
              setCompareFor(sameShape(result, photo) ? result.id : null)
              onOpen(result)
            }}
          />
        ) : (
        <dl>
          {origin && (
            <>
              <dt>Made by Pixora</dt>
              <dd>
                <OriginBadge origin={origin} />{' '}
                {creation && (
                  <>
                    {OPERATIONS[creation.operation] ?? creation.operation}
                    {creation.params.style && ` · ${creation.params.style} style`}
                    {creation.params.person_name && ` · ${creation.params.person_name}`}
                    {creation.params.prompt && <div className="muted">“{creation.params.prompt}”</div>}
                    {creation.operation !== 'remove_background' &&
                      creation.operation !== 'enhance' &&
                      creation.operation !== 'restore' && <div className="muted">Seed {creation.params.seed}</div>}
                    <div className="row creation-actions">
                      {creation.source && (
                        <button className="link" onClick={() => onOpen(creation.source!)}>
                          Open the original
                        </button>
                      )}
                      {creation.kind === 'generated' && (
                        <button className="link" onClick={() => onReuse(creation)}>
                          Use these settings again
                        </button>
                      )}
                    </div>
                  </>
                )}
                <div className="row creation-actions">
                  {confirmDelete ? (
                    <>
                      <span>Delete this picture?</span>
                      <button className="danger" disabled={deleting} onClick={remove}>
                        {deleting ? 'Deleting…' : 'Yes, delete'}
                      </button>
                      <button disabled={deleting} onClick={() => setDeleteFor(null)}>
                        Keep
                      </button>
                    </>
                  ) : (
                    <button className="link danger-link" onClick={() => (setDeleteError(null), setDeleteFor(photo.id))}>
                      Delete this picture
                    </button>
                  )}
                </div>
                {confirmDelete && (
                  <div className="muted">
                    It leaves the library and its file goes to the Recycle Bin. Your own photos are not touched.
                  </div>
                )}
                {confirmDelete && deleteError && <div className="form-error">{deleteError}</div>}
              </dd>
            </>
          )}
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
        )}
      </aside>
    </div>
  )
}
