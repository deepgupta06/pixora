import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import { saveBlob } from '../saveFile'
import { UnlockDialog, type Opened } from './Locked'

/** A picture the service holds in memory for the user, not saved anywhere. */
export interface Held {
  job_id: number
  index: number
}

const heldKey = (r: Held) => `${r.job_id}:${r.index}`
const THUMB = 256

/** One picture in the viewer: either still held by the service, or opened from a locked file. */
interface Item {
  id: string
  url: string
  thumb: string
  held?: Held
  /** The bytes of a picture opened from a locked file, so a normal copy can be made of it. */
  blob?: Blob
}

function forget(item: Item) {
  URL.revokeObjectURL(item.url)
  if (item.thumb !== item.url) URL.revokeObjectURL(item.thumb)
}

interface Props {
  incognito: boolean
  held: Held[]
  error: string | null
  onSave: (picture: Held) => void
  onLock: (pictures: Held[]) => void
  onDiscard: (pictures: Held[]) => void
}

/**
 * The pictures made in incognito mode, one large with Previous and Next, and all of them as
 * thumbnails below. Pictures opened from a locked file join them, in memory only, and are
 * forgotten when this screen is left.
 */
export function IncognitoViewer({ incognito, held, error, onSave, onLock, onDiscard }: Props) {
  const [opened, setOpened] = useState<Item[]>([])
  const [file, setFile] = useState<File | null>(null)
  const [current, setCurrent] = useState(0)
  const [picked, setPicked] = useState<ReadonlySet<string>>(new Set())
  const [maximized, setMaximized] = useState(false)
  const [note, setNote] = useState<string | null>(null)
  const picker = useRef<HTMLInputElement>(null)
  const nextId = useRef(1)

  const items = useMemo<Item[]>(
    () => [
      ...held.map((r) => ({
        id: `h${heldKey(r)}`,
        url: api.heldUrl(r.job_id, r.index),
        thumb: api.heldUrl(r.job_id, r.index, THUMB),
        held: r,
      })),
      ...opened,
    ],
    [held, opened],
  )
  const n = items.length
  const index = Math.min(current, n - 1)
  const item: Item | undefined = items[index]
  const chosen = held.filter((r) => picked.has(heldKey(r)))

  // A picture that has just arrived is the one to look at.
  const seen = useRef<ReadonlySet<string>>(new Set())
  useEffect(() => {
    const fresh = items.findIndex((it) => !seen.current.has(it.id))
    seen.current = new Set(items.map((it) => it.id))
    if (fresh < 0) return
    const id = setTimeout(() => setCurrent(fresh), 0)
    return () => clearTimeout(id)
  }, [items])

  // Pictures opened from a locked file go when the user leaves this screen.
  const latest = useRef<Item[]>([])
  useEffect(() => {
    latest.current = opened
  }, [opened])
  useEffect(
    () => () => {
      latest.current.forEach(forget)
    },
    [],
  )

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (document.querySelector('dialog[open]')) return // a dialog has the keyboard
      const target = e.target
      if (
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        (target instanceof HTMLInputElement && target.type !== 'checkbox')
      )
        return
      if (e.key === 'ArrowLeft' && n > 1) setCurrent((c) => (Math.min(c, n - 1) + n - 1) % n)
      else if (e.key === 'ArrowRight' && n > 1) setCurrent((c) => (Math.min(c, n - 1) + 1) % n)
      else if (e.key === 'Escape') setMaximized(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [n])

  const go = (step: number) => setCurrent((c) => (Math.min(c, n - 1) + step + n) % n)

  const tick = (r: Held, on: boolean) => {
    const next = new Set(picked)
    if (on) next.add(heldKey(r))
    else next.delete(heldKey(r))
    setPicked(next)
  }

  const forgetOpened = (id?: string) => {
    const gone = opened.filter((o) => id === undefined || o.id === id)
    gone.forEach(forget)
    setOpened(opened.filter((o) => !gone.includes(o)))
  }

  const exportOne = (it: Item) => {
    if (!it.blob) return
    setNote(null)
    saveBlob(it.blob, { name: 'picture.png', description: 'PNG picture', mime: 'image/png', ext: '.png' }).catch(
      (e: unknown) => setNote(e instanceof Error ? e.message : 'The picture could not be saved'),
    )
  }

  const add = (pictures: Opened[]) =>
    setOpened((before) => [
      ...before,
      ...pictures.map((p) => ({ id: `o${nextId.current++}`, url: p.url, thumb: p.thumb, blob: p.blob })),
    ])

  // Nothing to show, and not in incognito mode: the viewer is not here.
  if (!incognito && n === 0 && !file) return null

  const frozen = { onContextMenu: (e: { preventDefault: () => void }) => e.preventDefault() }

  return (
    <section className={`inc ${maximized ? 'maximized' : ''}`} aria-label="Incognito pictures">
      <div className="row inc-head">
        <h3>Incognito pictures{n > 0 ? ` (${n})` : ''}</h3>
        <div className="row inc-tools">
          {held.length > 0 && (
            <>
              <button onClick={() => setPicked(chosen.length === held.length ? new Set() : new Set(held.map(heldKey)))}>
                {chosen.length === held.length ? 'Select none' : 'Select all'}
              </button>
              <button
                className="primary"
                disabled={chosen.length === 0}
                onClick={() => onLock(chosen)}
                title={
                  chosen.length === 0
                    ? 'Tick the pictures to lock together first'
                    : 'Lock the ticked pictures together in one file, under one key'
                }
              >
                Lock selected{chosen.length > 0 ? ` (${chosen.length})` : ''}…
              </button>
            </>
          )}
          <button onClick={() => picker.current?.click()}>Open locked pictures…</button>
          {opened.length > 0 && <button onClick={() => forgetOpened()}>Close locked pictures</button>}
          <button onClick={() => setMaximized(!maximized)} aria-pressed={maximized}>
            {maximized ? 'Restore' : 'Maximize'}
          </button>
        </div>
      </div>
      <input
        ref={picker}
        type="file"
        accept=".pxlock"
        hidden
        onChange={(e) => {
          const chosenFile = e.target.files?.[0]
          e.target.value = '' // so the same file can be chosen again
          if (chosenFile) setFile(chosenFile)
        }}
      />
      {file && <UnlockDialog file={file} onClose={() => setFile(null)} onOpened={add} />}

      <div className="inc-stage" {...frozen}>
        {item ? (
          <img src={item.url} alt="" draggable={false} />
        ) : (
          <p className="muted">
            Pictures you make in incognito mode appear here, held in memory only. A locked file can be opened here too.
          </p>
        )}
        <button className="inc-prev" aria-label="Previous picture" disabled={n < 2} onClick={() => go(-1)}>
          ‹
        </button>
        <button className="inc-next" aria-label="Next picture" disabled={n < 2} onClick={() => go(1)}>
          ›
        </button>
        {n > 0 && (
          <span className="inc-count">
            {index + 1} of {n}
          </span>
        )}
      </div>

      {item && (
        <div className="row inc-actions">
          {item.held ? (
            <>
              <label className="check inc-pick">
                <input
                  type="checkbox"
                  checked={picked.has(heldKey(item.held))}
                  onChange={(e) => tick(item.held!, e.target.checked)}
                />
                Select
              </label>
              <button className="primary" onClick={() => onSave(item.held!)}>
                Save…
              </button>
              <button onClick={() => onDiscard([item.held!])}>Discard</button>
              {held.length > 1 && <button onClick={() => onDiscard(held)}>Discard all</button>}
            </>
          ) : (
            <>
              <button onClick={() => exportOne(item)} title="Make a normal copy that any program can open">
                Export as normal image…
              </button>
              <button onClick={() => forgetOpened(item.id)}>Close this picture</button>
            </>
          )}
        </div>
      )}

      {held.length > 1 && (
        <p className="muted inc-hint">Tick the pictures to lock together, then press Lock selected: one key opens them all.</p>
      )}
      {n > 0 && (
        <div className="inc-strip" role="list" {...frozen}>
          {items.map((it, i) => (
            <div key={it.id} className="inc-cell" role="listitem">
              {it.held && (
                <label className="held-pick">
                  <input
                    type="checkbox"
                    checked={picked.has(heldKey(it.held))}
                    aria-label="Lock this picture with the others ticked"
                    onChange={(e) => tick(it.held!, e.target.checked)}
                  />
                </label>
              )}
              <button
                className="inc-thumb"
                aria-current={i === index ? 'true' : undefined}
                aria-label={`Picture ${i + 1} of ${n}`}
                onClick={() => setCurrent(i)}
              >
                <img src={it.thumb} alt="" draggable={false} />
                {!it.held && (
                  <span className="inc-badge" title="From a locked file">
                    🔒
                  </span>
                )}
              </button>
            </div>
          ))}
        </div>
      )}

      <p className="muted">
        Held in memory only. Save puts a picture in a folder you choose; it is not added to Pixora. Lock selected puts
        the ticked pictures in one encrypted file under one key, shown once, that only Pixora can open. Pictures opened
        from a locked file are forgotten when you leave this screen.
      </p>
      {(error || note) && <p className="form-error">{error ?? note}</p>}
    </section>
  )
}
