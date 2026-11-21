import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
  type CSSProperties,
  type Dispatch,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from 'react'
import { api } from '../api'
import { LockedContext, type Locked } from '../lockedContext'
import { base64ToBlob, fileToBase64, saveBlob } from '../saveFile'

/**
 * The key to a locked file, shown once. Pixora keeps no copy of it anywhere, so a key that
 * is lost is a set of pictures that is lost.
 */
export function KeyDialog({ value, count, onDone }: { value: string; count: number; onDone: () => void }) {
  const ref = useRef<HTMLDialogElement>(null)
  const [saved, setSaved] = useState(false)
  const [copied, setCopied] = useState<'yes' | 'failed' | null>(null)

  useEffect(() => {
    const dialog = ref.current
    if (dialog && !dialog.open) dialog.showModal()
  }, [])

  // Leaving the page now would lose the key for good.
  useEffect(() => {
    if (saved) return
    const warn = (e: BeforeUnloadEvent) => e.preventDefault()
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [saved])

  const groups = value.split('-')
  const lines = [0, 4, 8, 12].map((i) => groups.slice(i, i + 4).join('-')).filter(Boolean)

  return (
    <dialog
      ref={ref}
      className="dialog lock-dialog"
      aria-label="Key for the locked pictures"
      // Escape must not throw the key away.
      onCancel={(e) => e.preventDefault()}
    >
      <header>
        <h2>Your key</h2>
      </header>
      <p>
        {count === 1 ? 'The picture is' : `The ${count} pictures are`} locked in the file you saved. This key is the only
        thing that opens {count === 1 ? 'it' : 'them'}.
      </p>
      <pre className="lock-key">{lines.join('\n')}</pre>
      <div className="row lock-copy">
        <button
          onClick={() =>
            void navigator.clipboard.writeText(value).then(
              () => setCopied('yes'),
              () => setCopied('failed'),
            )
          }
        >
          Copy key
        </button>
        {copied === 'yes' && <span className="muted">Copied</span>}
        {copied === 'failed' && <span className="form-error">Could not copy. Select the key and copy it yourself.</span>}
      </div>
      <p className="lock-warning">
        Pixora does not keep this key and cannot recover it. If you lose it, {count === 1 ? 'the picture is' : 'the pictures are'}{' '}
        gone for good, for you and for everyone. Put it in a password manager or write it down before you go on.
      </p>
      <label className="check">
        <input type="checkbox" checked={saved} onChange={(e) => setSaved(e.target.checked)} />I have saved the key
        somewhere safe
      </label>
      <div className="row lock-actions">
        <button className="primary" disabled={!saved} onClick={onDone}>
          Done
        </button>
      </div>
    </dialog>
  )
}

// ---- opening a locked file ------------------------------------------------------------

/** A picture out of a locked file. It exists only here, in memory, until its window closes. */
interface Picture {
  blob: Blob
  url: string
}

interface Win {
  id: number
  pictures: Picture[]
  x: number
  y: number
  w: number
  h: number
  maximized: boolean
  z: number
  /** The picture being looked at on its own, or null for all of them at once. */
  shown: number | null
}

interface Desk {
  windows: Win[]
  next: number
  top: number
}

type Action =
  | { type: 'open'; pictures: Picture[] }
  | { type: 'close'; id: number }
  | { type: 'front'; id: number }
  | { type: 'move'; id: number; x: number; y: number }
  | { type: 'size'; id: number; w: number; h: number }
  | { type: 'maximize'; id: number }
  | { type: 'show'; id: number; shown: number | null }
  | { type: 'fit' }

const clamp = (value: number, low: number, high: number) => Math.min(Math.max(value, low), Math.max(low, high))

/** Let go of a window's pictures, so the browser keeps nothing of them. */
function forget(win: Win | undefined) {
  win?.pictures.forEach((p) => URL.revokeObjectURL(p.url))
}

/**
 * Every window is kept wholly inside the Pixora window. Dragging, growing and shrinking the
 * app can never carry one off the screen it is on.
 */
function desk(state: Desk, action: Action): Desk {
  const vw = window.innerWidth
  const vh = window.innerHeight
  const change = (id: number, f: (win: Win) => Win): Desk => ({
    ...state,
    windows: state.windows.map((win) => (win.id === id ? f(win) : win)),
  })
  switch (action.type) {
    case 'open': {
      const w = Math.min(920, Math.round(vw * 0.72))
      const h = Math.min(660, Math.round(vh * 0.72))
      const step = (state.next % 8) * 30 // each new window a little further along, so none hides another
      const win: Win = {
        id: state.next,
        pictures: action.pictures,
        w,
        h,
        x: clamp(70 + step, 0, vw - w),
        y: clamp(70 + step, 0, vh - h),
        maximized: false,
        z: state.top + 1,
        shown: action.pictures.length === 1 ? 0 : null,
      }
      return { windows: [...state.windows, win], next: state.next + 1, top: state.top + 1 }
    }
    case 'close':
      return { ...state, windows: state.windows.filter((win) => win.id !== action.id) }
    case 'front':
      if (state.windows.find((win) => win.id === action.id)?.z === state.top) return state
      return { ...change(action.id, (win) => ({ ...win, z: state.top + 1 })), top: state.top + 1 }
    case 'move':
      return change(action.id, (win) => ({
        ...win,
        x: clamp(action.x, 0, vw - win.w),
        y: clamp(action.y, 0, vh - win.h),
      }))
    case 'size':
      return change(action.id, (win) =>
        win.maximized || (win.w === action.w && win.h === action.h) ? win : { ...win, w: action.w, h: action.h },
      )
    case 'maximize':
      return change(action.id, (win) => ({ ...win, maximized: !win.maximized }))
    case 'show':
      return change(action.id, (win) => ({ ...win, shown: action.shown }))
    case 'fit':
      return {
        ...state,
        windows: state.windows.map((win) => {
          const w = Math.min(win.w, vw)
          const h = Math.min(win.h, vh)
          return { ...win, w, h, x: clamp(win.x, 0, vw - w), y: clamp(win.y, 0, vh - h) }
        }),
      }
  }
}

/**
 * Locked files open in windows that belong to the app as a whole, so they stay while the user
 * moves between views. Nothing of what is in them is written anywhere: it is in memory here,
 * and goes when its window is closed.
 */
export function LockedProvider({ children }: { children: ReactNode }) {
  const [asking, setAsking] = useState<File | null>(null)
  const [{ windows, top }, dispatch] = useReducer(desk, { windows: [], next: 1, top: 0 })
  const latest = useRef<Win[]>([])
  useEffect(() => {
    latest.current = windows
  }, [windows])

  const close = useCallback((id: number) => {
    forget(latest.current.find((win) => win.id === id))
    dispatch({ type: 'close', id })
  }, [])
  const closeAll = () => {
    latest.current.forEach(forget)
    latest.current.forEach((win) => dispatch({ type: 'close', id: win.id }))
  }

  // If the provider itself goes, so do the pictures.
  useEffect(
    () => () => {
      latest.current.forEach(forget)
    },
    [],
  )

  useEffect(() => {
    const onResize = () => dispatch({ type: 'fit' })
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  useEffect(() => {
    if (windows.length === 0) return
    const onKey = (e: KeyboardEvent) => {
      if (document.querySelector('dialog[open]')) return // a dialog has the keyboard
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      const front = windows.reduce((a, b) => (b.z > a.z ? b : a))
      const n = front.pictures.length
      if (e.key === 'Escape') {
        if (front.shown !== null && n > 1) dispatch({ type: 'show', id: front.id, shown: null })
        else close(front.id)
      } else if (front.shown !== null && n > 1 && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
        const step = e.key === 'ArrowLeft' ? -1 : 1
        dispatch({ type: 'show', id: front.id, shown: (front.shown + step + n) % n })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [windows, close])

  const locked = useMemo<Locked>(() => ({ openFile: setAsking }), [])

  return (
    <LockedContext.Provider value={locked}>
      {children}
      {asking && (
        <UnlockDialog
          file={asking}
          onClose={() => setAsking(null)}
          onOpened={(pictures) => dispatch({ type: 'open', pictures })}
        />
      )}
      {windows.map((win) => (
        <LockedWindow key={win.id} win={win} dispatch={dispatch} onClose={close} />
      ))}
      {windows.length > 1 && (
        <button className="lock-close-all" onClick={closeAll} style={{ zIndex: 101 + top }}>
          Close all locked pictures ({windows.length})
        </button>
      )}
    </LockedContext.Provider>
  )
}

/** Asks for the key to a file, and opens it. The key is kept only while it is being typed. */
function UnlockDialog({
  file,
  onClose,
  onOpened,
}: {
  file: File
  onClose: () => void
  onOpened: (pictures: Picture[]) => void
}) {
  const ref = useRef<HTMLDialogElement>(null)
  const [key, setKey] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const dialog = ref.current
    if (dialog && !dialog.open) dialog.showModal()
  }, [])

  // Closed here, not left to the dialog's own close event, which the browser may hold back.
  const finish = () => {
    ref.current?.close()
    onClose()
  }

  const unlock = async () => {
    setBusy(true)
    setError(null)
    try {
      const sent = await api.openLocked(await fileToBase64(file), key)
      const pictures = sent.images.map((data) => {
        const blob = base64ToBlob(data, 'image/png')
        return { blob, url: URL.createObjectURL(blob) }
      })
      setKey('') // not kept a moment longer than it was needed
      onOpened(pictures)
      finish()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'That did not work')
    } finally {
      setBusy(false)
    }
  }

  return (
    <dialog ref={ref} className="dialog lock-dialog" aria-label="Open locked pictures" onClose={onClose}>
      <header>
        <h2>Open locked pictures</h2>
        <button onClick={finish} aria-label="Close">
          ✕
        </button>
      </header>
      <form
        onSubmit={(e) => {
          e.preventDefault()
          if (key.trim() && !busy) void unlock()
        }}
      >
        <p className="muted">{file.name}</p>
        <label className="field">
          <span>Key</span>
          <input
            value={key}
            onChange={(e) => setKey(e.target.value)}
            placeholder="ABCD-EFGH-IJKL-MNOP-…"
            autoComplete="off"
            autoCapitalize="characters"
            spellCheck={false}
            autoFocus
            className="lock-key-input"
          />
        </label>
        {error && <p className="form-error">{error}</p>}
        <div className="row lock-actions">
          <button type="submit" className="primary" disabled={busy || !key.trim()}>
            {busy ? 'Opening…' : 'Open'}
          </button>
        </div>
      </form>
    </dialog>
  )
}

/**
 * One locked file: all its pictures at once, or one picture large. The window can be dragged,
 * resized and maximized, but never leaves the Pixora window. The pictures cannot be dragged
 * out of it or saved from a menu; a normal copy is made only by the Export button.
 */
const LockedWindow = memo(function LockedWindow({
  win,
  dispatch,
  onClose,
}: {
  win: Win
  dispatch: Dispatch<Action>
  onClose: (id: number) => void
}) {
  const el = useRef<HTMLDivElement>(null)
  const [note, setNote] = useState<string | null>(null)
  const n = win.pictures.length
  const shown = win.shown

  // What the user makes of the window by its corner is remembered, so it comes back as left.
  useEffect(() => {
    const node = el.current
    if (!node || win.maximized) return
    const watch = new ResizeObserver(() =>
      dispatch({ type: 'size', id: win.id, w: node.offsetWidth, h: node.offsetHeight }),
    )
    watch.observe(node)
    return () => watch.disconnect()
  }, [win.id, win.maximized, dispatch])

  const drag = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (win.maximized || e.button !== 0 || (e.target as HTMLElement).closest('button')) return
    const from = { x: e.clientX - win.x, y: e.clientY - win.y }
    const move = (ev: PointerEvent) => dispatch({ type: 'move', id: win.id, x: ev.clientX - from.x, y: ev.clientY - from.y })
    const stop = () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', stop)
      window.removeEventListener('pointercancel', stop)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', stop)
    window.addEventListener('pointercancel', stop)
    try {
      e.currentTarget.setPointerCapture(e.pointerId) // keep following the pointer past the window's edge
    } catch {
      // the pointer events on the page are enough
    }
  }

  const show = (index: number | null) => dispatch({ type: 'show', id: win.id, shown: index })
  const exportOne = (index: number) => {
    setNote(null)
    saveBlob(win.pictures[index].blob, {
      name: 'picture.png',
      description: 'PNG picture',
      mime: 'image/png',
      ext: '.png',
    }).catch((e: unknown) => setNote(e instanceof Error ? e.message : 'The picture could not be saved'))
  }

  const style: CSSProperties = win.maximized
    ? { zIndex: 100 + win.z }
    : {
        zIndex: 100 + win.z,
        left: win.x,
        top: win.y,
        width: win.w,
        height: win.h,
        // Growing it by the corner stops at the edge of the window the app is in.
        maxWidth: `calc(100vw - ${win.x}px)`,
        maxHeight: `calc(100vh - ${win.y}px)`,
      }

  return (
    <div
      ref={el}
      className={`lock-win ${win.maximized ? 'maximized' : ''}`}
      style={style}
      role="dialog"
      aria-label="Locked pictures"
      onPointerDown={() => dispatch({ type: 'front', id: win.id })}
      onContextMenu={(e) => e.preventDefault()}
      onDragStart={(e) => e.preventDefault()}
    >
      <div className="lock-win-bar" onPointerDown={drag} onDoubleClick={() => dispatch({ type: 'maximize', id: win.id })}>
        <span className="lock-win-title">Locked pictures{n > 1 ? ` · ${n}` : ''}</span>
        <span className="lock-win-buttons">
          {shown !== null && (
            <button onClick={() => exportOne(shown)} title="Make a normal copy that any program can open">
              Export as normal image…
            </button>
          )}
          <button
            onClick={() => dispatch({ type: 'maximize', id: win.id })}
            aria-label={win.maximized ? 'Restore' : 'Maximize'}
            title={win.maximized ? 'Restore' : 'Maximize'}
          >
            {win.maximized ? '❐' : '□'}
          </button>
          <button onClick={() => onClose(win.id)} aria-label="Close" title="Close, and forget the pictures">
            ✕
          </button>
        </span>
      </div>
      <div className="lock-win-body">
        {shown === null ? (
          <div className="lock-grid">
            {win.pictures.map((p, i) => (
              <button key={p.url} className="lock-tile" onClick={() => show(i)} aria-label={`Picture ${i + 1} of ${n}`}>
                <img src={p.url} alt="" draggable={false} />
              </button>
            ))}
          </div>
        ) : (
          <div className="lock-light">
            <img src={win.pictures[shown].url} alt="" draggable={false} />
            {n > 1 && (
              <>
                <button className="lock-prev" aria-label="Previous picture" onClick={() => show((shown - 1 + n) % n)}>
                  ‹
                </button>
                <button className="lock-next" aria-label="Next picture" onClick={() => show((shown + 1) % n)}>
                  ›
                </button>
                <div className="lock-light-bar">
                  <button onClick={() => show(null)}>All pictures</button>
                  <span className="muted">
                    {shown + 1} of {n}
                  </span>
                </div>
              </>
            )}
          </div>
        )}
      </div>
      {note && <p className="form-error lock-win-note">{note}</p>}
    </div>
  )
})
