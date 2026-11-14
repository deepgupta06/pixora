import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { api, type PeopleOverview, type Person, type Question } from '../api'

interface Props {
  /** Photos checked for faces so far; a change means new faces may have arrived. */
  facesDone: number
  /** Why faces are not being found, when they are not. */
  note: string | null
  onShowPerson: (personId: number) => void
}

export function PeopleView({ facesDone, note, onShowPerson }: Props) {
  const [overview, setOverview] = useState<PeopleOverview | null>(null)
  const [question, setQuestion] = useState<Question | null>(null)

  const reload = useCallback(async () => {
    const [nextOverview, nextQuestion] = await Promise.all([api.people(), api.question()])
    setOverview(nextOverview)
    setQuestion(nextQuestion)
  }, [])

  useEffect(() => {
    const id = setTimeout(() => void reload().catch(() => {}), 0)
    return () => clearTimeout(id)
  }, [reload, facesDone])

  const visible = overview?.people.filter((p) => !p.hidden) ?? []
  const hidden = overview?.people.filter((p) => p.hidden) ?? []
  const nothingYet = overview !== null && overview.people.length === 0 && question?.type === 'done'

  return (
    <main className="people">
      {question && question.type !== 'done' && (
        // Keyed by the faces asked about, so each new question starts clean.
        <QuestionCard key={question.face_ids.join(',')} question={question} people={overview?.people ?? []} onAnswered={reload} />
      )}

      {nothingYet && (
        <div className="empty">
          <h2>No people yet</h2>
          <p className="muted">
            {note ?? 'People appear here once faces have been found in at least three photos of the same person.'}
          </p>
        </div>
      )}

      {visible.length > 0 && (
        <>
          <h3>
            {visible.length} {visible.length === 1 ? 'person' : 'people'}
            {overview && overview.unnamed_groups > 0 && ` · ${overview.unnamed_groups} more to name`}
          </h3>
          <div className="people-grid">
            {visible.map((p) => (
              <PersonCard key={p.id} person={p} onShow={() => onShowPerson(p.id)} onChanged={reload} />
            ))}
          </div>
        </>
      )}

      {hidden.length > 0 && (
        <>
          <h3 style={{ marginTop: 28 }}>Hidden</h3>
          <div className="people-grid">
            {hidden.map((p) => (
              <PersonCard key={p.id} person={p} onShow={() => onShowPerson(p.id)} onChanged={reload} />
            ))}
          </div>
        </>
      )}
    </main>
  )
}

function QuestionCard({
  question,
  people,
  onAnswered,
}: {
  question: Question
  people: Person[]
  onAnswered: () => Promise<void>
}) {
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const answer = async (call: () => Promise<unknown>) => {
    setBusy(true)
    setError(null)
    try {
      await call()
      await onAnswered()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'That did not work')
      setBusy(false)
    }
  }

  const submitName = (e: FormEvent) => {
    e.preventDefault()
    if (name.trim()) void answer(() => api.nameFaces(question.face_ids, name))
  }

  const count = question.face_ids.length
  const where = `${count.toLocaleString()} ${count === 1 ? 'face' : 'faces'} in ${question.photo_count.toLocaleString()} ${question.photo_count === 1 ? 'photo' : 'photos'}`

  return (
    <section className="question" aria-live="polite">
      <h2>{question.type === 'name' ? 'Who is this?' : `Is this also ${question.name}?`}</h2>
      <p className="muted">{where}</p>
      <div className="question-faces">
        {question.sample_face_ids.map((id) => (
          <img key={id} src={api.faceUrl(id)} alt="" />
        ))}
      </div>
      {question.type === 'name' ? (
        <form className="question-actions" onSubmit={submitName}>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Name"
            aria-label="Name"
            list="known-people"
            autoFocus
          />
          <datalist id="known-people">
            {people.map((p) => (
              <option key={p.id} value={p.name} />
            ))}
          </datalist>
          <button className="primary" disabled={busy || !name.trim()}>
            Save
          </button>
          <button type="button" disabled={busy} onClick={() => void answer(() => api.dismissFaces(question.face_ids))}>
            Skip
          </button>
        </form>
      ) : (
        <div className="question-actions">
          <button
            className="primary"
            disabled={busy}
            onClick={() => void answer(() => api.confirmFaces(question.person_id!, question.face_ids, true))}
          >
            Yes
          </button>
          <button
            disabled={busy}
            onClick={() => void answer(() => api.confirmFaces(question.person_id!, question.face_ids, false))}
          >
            No
          </button>
        </div>
      )}
      {error && <p className="form-error">{error}</p>}
    </section>
  )
}

function PersonCard({ person, onShow, onChanged }: { person: Person; onShow: () => void; onChanged: () => Promise<void> }) {
  const [editing, setEditing] = useState(false)
  const [name, setName] = useState(person.name)
  const [error, setError] = useState<string | null>(null)

  const change = async (update: { name?: string; hidden?: boolean }) => {
    setError(null)
    try {
      await api.updatePerson(person.id, update)
      setEditing(false)
      await onChanged()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'That did not work')
    }
  }

  return (
    <div className="person">
      <button className="person-cover" onClick={onShow} aria-label={`Show photos of ${person.name}`}>
        {person.cover_face_id !== null && <img src={api.faceUrl(person.cover_face_id)} alt="" loading="lazy" />}
      </button>
      {editing ? (
        <form
          onSubmit={(e) => {
            e.preventDefault()
            if (name.trim()) void change({ name })
          }}
        >
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === 'Escape' && setEditing(false)}
            aria-label={`New name for ${person.name}`}
            autoFocus
          />
        </form>
      ) : (
        <div className="person-name">{person.name}</div>
      )}
      <div className="person-row muted">
        <span>
          {person.photo_count.toLocaleString()} {person.photo_count === 1 ? 'photo' : 'photos'}
        </span>
        <span>
          <button className="link" onClick={() => (setName(person.name), setEditing(!editing))}>
            {editing ? 'Cancel' : 'Rename'}
          </button>{' '}
          <button className="link" onClick={() => void change({ hidden: !person.hidden })}>
            {person.hidden ? 'Show' : 'Hide'}
          </button>
        </span>
      </div>
      {error && <p className="form-error">{error}</p>}
    </div>
  )
}
