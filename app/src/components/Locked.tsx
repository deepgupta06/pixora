import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { base64ToBlob, fileToBase64, saveBlob, type SaveOptions, type SaveResult } from '../saveFile'

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

/** A picture out of a locked file. It exists only here, in memory, until it is let go. */
export interface Opened {
  /** The picture, as PNG. */
  blob: Blob
  /** Addresses in this page's memory, for the picture and a small version of it. */
  url: string
  thumb: string
}

/** Asks for the key to a locked file, and opens it. The key is kept only while it is being typed. */
export function UnlockDialog({
  file,
  onClose,
  onOpened,
}: {
  file: File
  onClose: () => void
  onOpened: (pictures: Opened[]) => void
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
      const pictures = sent.images.map((data, i) => {
        const blob = base64ToBlob(data, 'image/png')
        const url = URL.createObjectURL(blob)
        const small = sent.thumbs[i]
        return { blob, url, thumb: small ? URL.createObjectURL(base64ToBlob(small, 'image/jpeg')) : url }
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
        <p className="muted">One key opens every picture in this file.</p>
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
 * Asks for the key of a locked file that pictures are to be added to. `onAdd` makes the new
 * file and the caller puts it somewhere; the key is kept only while it is being typed.
 */
export function AddDialog({
  file,
  count,
  onClose,
  onAdd,
}: {
  file: File
  count: number
  onClose: () => void
  /** Refuses, with a message to show, for a wrong key or a damaged file. */
  onAdd: (key: string) => Promise<void>
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

  const add = async () => {
    setBusy(true)
    setError(null)
    try {
      await onAdd(key)
      setKey('') // not kept a moment longer than it was needed
      finish()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'That did not work')
    } finally {
      setBusy(false)
    }
  }

  return (
    <dialog
      ref={ref}
      className="dialog lock-dialog"
      aria-label="Add to a locked file"
      // Once the file is being made it goes on to be saved, so the dialog stays until that is handed over.
      onCancel={(e) => {
        if (busy) e.preventDefault()
      }}
      onClose={onClose}
    >
      <header>
        <h2>Add to a locked file</h2>
        <button onClick={finish} disabled={busy} aria-label="Close">
          ✕
        </button>
      </header>
      <form
        onSubmit={(e) => {
          e.preventDefault()
          if (key.trim() && !busy) void add()
        }}
      >
        <p className="muted">{file.name}</p>
        <p className="muted">
          {count === 1 ? 'The picture you ticked goes' : `The ${count} pictures you ticked go`} into this file under its
          own key, so one key opens everything in it. No new key is made.
        </p>
        <label className="field">
          <span>Key of this file</span>
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
        <p className="muted">
          A Save window follows. Save over this file to keep one file, or give it a new name to keep this one as it is.
        </p>
        <div className="row lock-actions">
          <button type="submit" className="primary" disabled={busy || !key.trim()}>
            {busy ? 'Adding…' : count === 1 ? 'Add the picture' : `Add ${count} pictures`}
          </button>
        </div>
      </form>
    </dialog>
  )
}

/**
 * The browser opens its save window only for a click it saw a moment ago, and making a file can
 * take longer than that. This asks for a click, so a file that has been made is not lost for want
 * of one.
 */
export function SaveReadyDialog({
  blob,
  options,
  onDone,
}: {
  blob: Blob
  options: SaveOptions
  onDone: (result: Exclude<SaveResult, 'needs-click'>) => void
}) {
  const ref = useRef<HTMLDialogElement>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const dialog = ref.current
    if (dialog && !dialog.open) dialog.showModal()
  }, [])

  const save = async () => {
    setError(null)
    try {
      // Called straight from the click, so the browser lets its save window open.
      const result = await saveBlob(blob, options, true)
      if (result === 'written' || result === 'handed-over') onDone(result)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The file could not be saved')
    }
  }

  return (
    <dialog
      ref={ref}
      className="dialog lock-dialog"
      aria-label="Save the file"
      onCancel={(e) => {
        e.preventDefault()
        onDone('cancelled')
      }}
    >
      <header>
        <h2>Your file is ready</h2>
      </header>
      <p>
        It took a moment to make, so your browser wants a fresh click before it opens its Save window. If you cancel
        now, nothing is saved and the pictures stay held.
      </p>
      {error && <p className="form-error">{error}</p>}
      <div className="row lock-actions">
        <button onClick={() => onDone('cancelled')}>Cancel</button>
        <button className="primary" onClick={() => void save()}>
          Choose where to save…
        </button>
      </div>
    </dialog>
  )
}
