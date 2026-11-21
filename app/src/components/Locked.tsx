import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { base64ToBlob, fileToBase64, saveBlob } from '../saveFile'

/**
 * The key to a locked picture, shown once. Pixora keeps no copy of it anywhere, so a key
 * that is lost is a picture that is lost.
 */
export function KeyDialog({ value, onDone }: { value: string; onDone: () => void }) {
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
      aria-label="Key for the locked picture"
      // Escape must not throw the key away.
      onCancel={(e) => e.preventDefault()}
    >
      <header>
        <h2>Your key</h2>
      </header>
      <p>The picture is locked in the file you saved. This key is the only thing that opens it.</p>
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
        Pixora does not keep this key and cannot recover it. If you lose it, the picture is gone for good, for you and
        for everyone. Put it in a password manager or write it down before you go on.
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

interface Opened {
  blob: Blob
  url: string
  settings: Record<string, unknown>
}

function Detail({ label, value }: { label: string; value: unknown }) {
  if (value === undefined || value === null || value === '') return null
  return (
    <>
      <dt>{label}</dt>
      <dd>{String(value)}</dd>
    </>
  )
}

/**
 * A locked picture: ask for its key, then show it. The picture is decrypted by the service in
 * memory, handed over so the browser does not keep a copy, and forgotten when this closes.
 */
export function OpenLocked({ file, onClose }: { file: File; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null)
  const [key, setKey] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [opened, setOpened] = useState<Opened | null>(null)
  const url = opened?.url

  useEffect(() => {
    const dialog = ref.current
    if (dialog && !dialog.open) dialog.showModal()
  }, [])

  // The picture goes with this window.
  useEffect(
    () => () => {
      if (url) URL.revokeObjectURL(url)
    },
    [url],
  )

  const unlock = async () => {
    setBusy(true)
    setError(null)
    try {
      const sent = await api.openLocked(await fileToBase64(file), key)
      const blob = base64ToBlob(sent.image, 'image/png')
      setKey('') // not kept a moment longer than it was needed
      setOpened({ blob, url: URL.createObjectURL(blob), settings: sent.settings })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'That did not work')
    } finally {
      setBusy(false)
    }
  }

  const exportCopy = () =>
    opened &&
    saveBlob(opened.blob, { name: 'picture.png', description: 'PNG picture', mime: 'image/png', ext: '.png' }).catch(
      (e: unknown) => setError(e instanceof Error ? e.message : 'The picture could not be saved'),
    )

  return (
    <dialog
      ref={ref}
      className={`dialog lock-dialog ${opened ? 'lock-open' : ''}`}
      aria-label="Locked picture"
      onClose={onClose}
    >
      <header>
        <h2>{opened ? 'Locked picture' : 'Open a locked picture'}</h2>
        <button onClick={() => ref.current?.close()} aria-label="Close">
          ✕
        </button>
      </header>
      {opened ? (
        <>
          <img src={opened.url} alt="" />
          <dl className="lock-details">
            <Detail label="Description" value={opened.settings.prompt} />
            <Detail label="Person" value={opened.settings.person_name} />
            <Detail label="Look" value={opened.settings.style} />
            <Detail label="Seed" value={opened.settings.seed} />
          </dl>
          <p className="muted">
            Closing this forgets the picture. A normal copy can be opened by any program, so make one only when you mean
            to.
          </p>
          {error && <p className="form-error">{error}</p>}
          <div className="row lock-actions">
            <button onClick={() => void exportCopy()}>Export as normal image…</button>
            <button className="primary" onClick={() => ref.current?.close()}>
              Close
            </button>
          </div>
        </>
      ) : (
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
      )}
    </dialog>
  )
}
