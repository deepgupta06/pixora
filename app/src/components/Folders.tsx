import { useEffect, useRef, useState, type FormEvent } from 'react'
import { api, isDesktop, type Folder } from '../api'

interface FormProps {
  onAdded: (folders: Folder[]) => void
}

export function AddFolderForm({ onAdded }: FormProps) {
  const [path, setPath] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const add = async (value: string) => {
    if (!value.trim()) return
    setBusy(true)
    setError(null)
    try {
      onAdded(await api.addFolder(value))
      setPath('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not add that folder')
    } finally {
      setBusy(false)
    }
  }

  const submit = (e: FormEvent) => {
    e.preventDefault()
    void add(path)
  }

  // The native folder picker exists only in the desktop shell.
  const browse = async () => {
    const { open } = await import('@tauri-apps/plugin-dialog')
    const picked = await open({ directory: true, title: 'Choose a photo folder' })
    if (typeof picked === 'string') void add(picked)
  }

  return (
    <form className="add-folder" onSubmit={submit}>
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
        />
        <button className={isDesktop ? undefined : 'primary'} disabled={busy || !path.trim()}>
          Add folder
        </button>
      </div>
      {error && <p className="form-error">{error}</p>}
    </form>
  )
}

interface DialogProps {
  open: boolean
  onClose: () => void
  onChange: () => void
}

export function FoldersDialog({ open, onClose, onChange }: DialogProps) {
  const ref = useRef<HTMLDialogElement>(null)
  const [folders, setFolders] = useState<Folder[]>([])

  useEffect(() => {
    const dialog = ref.current
    if (!dialog) return
    if (open && !dialog.open) {
      dialog.showModal()
      api.folders().then(setFolders, () => setFolders([]))
    } else if (!open && dialog.open) {
      dialog.close()
    }
  }, [open])

  const update = (next: Folder[]) => {
    setFolders(next)
    onChange()
  }

  return (
    <dialog ref={ref} className="dialog" onClose={onClose}>
      <header>
        <h2>Library folders</h2>
        <button onClick={onClose} aria-label="Close">
          ✕
        </button>
      </header>
      <p className="muted">
        Pixora reads these folders and never changes or deletes anything in them.
      </p>
      <ul className="folder-list">
        {folders.map((f) => (
          <li key={f.id}>
            <div>
              <div className="folder-path">{f.path}</div>
              <div className={f.error ? 'form-error' : 'muted'}>
                {f.error ? `Not reachable: ${f.error}` : `${f.file_count.toLocaleString()} image files`}
              </div>
            </div>
            <button onClick={() => api.removeFolder(f.id).then(update)}>Remove</button>
          </li>
        ))}
        {folders.length === 0 && <li className="muted">No folders yet.</li>}
      </ul>
      <AddFolderForm onAdded={update} />
    </dialog>
  )
}
