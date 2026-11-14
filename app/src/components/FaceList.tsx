import { useState, type FormEvent } from 'react'
import { api, type PhotoFace } from '../api'

interface Props {
  faces: PhotoFace[]
  /** Called after a correction has been saved. */
  onChange: () => void
}

/** The people in one photo, with the corrections the user can make. */
export function FaceList({ faces, onChange }: Props) {
  return (
    <>
      {faces.map((face) => (
        <FaceRow key={`${face.id}-${face.person_id}`} face={face} onChange={onChange} />
      ))}
    </>
  )
}

function FaceRow({ face, onChange }: { face: PhotoFace; onChange: () => void }) {
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)

  const run = (call: () => Promise<unknown>) => {
    setBusy(true)
    void call().then(onChange, () => setBusy(false))
  }

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (name.trim()) run(() => api.assignFace(face.id, name))
  }

  return (
    <div className="face-row">
      <img src={api.faceUrl(face.id)} alt="" />
      {face.person_id !== null ? (
        <>
          <span className="face-name">{face.name}</span>
          <button className="link" disabled={busy} onClick={() => run(() => api.rejectFace(face.id))}>
            Not {face.name}
          </button>
        </>
      ) : (
        <form onSubmit={submit}>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder={face.suggested_name ? `${face.suggested_name}?` : 'Who is this?'}
            aria-label="Name this person"
          />
          {face.suggested_name && !name.trim() ? (
            <button type="button" disabled={busy} onClick={() => run(() => api.assignFace(face.id, face.suggested_name!))}>
              Yes
            </button>
          ) : (
            <button disabled={busy || !name.trim()}>Save</button>
          )}
        </form>
      )}
    </div>
  )
}
