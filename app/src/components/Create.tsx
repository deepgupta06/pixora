import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  api,
  type Creation,
  type PersonCandidate,
  type PhotoSummary,
  type PromptMention,
} from '../api'
import { useJob, useStudioStatus } from '../useStudio'
import { JobProgress, ModelsNeeded, Results } from './StudioParts'

interface Props {
  onOpen: (photo: PhotoSummary) => void
  /** The library gained pictures. */
  onChange: () => void
  /** Settings of an earlier picture to start from. */
  reuse: Creation | null
}

/** Who the picture should show: a person, nobody, or not decided yet. */
type Who = number | 'nobody' | null

function Face({ person, selected, onPick }: { person: PersonCandidate; selected: boolean; onPick: () => void }) {
  return (
    <button className="who" aria-pressed={selected} onClick={onPick}>
      {person.cover_face_id !== null && <img src={api.faceUrl(person.cover_face_id)} alt="" />}
      <span>
        {person.name}
        <span className="muted"> · {person.photo_count.toLocaleString()} photos</span>
      </span>
    </button>
  )
}

export function Create({ onOpen, onChange, reuse }: Props) {
  const [installed, setInstalled] = useState(0)
  const status = useStudioStatus(installed)

  const [prompt, setPrompt] = useState('')
  const [avoid, setAvoid] = useState('')
  const [size, setSize] = useState('square')
  const [style, setStyle] = useState('')
  const [count, setCount] = useState(1)
  const [seed, setSeed] = useState('')
  const [mentions, setMentions] = useState<PromptMention[]>([])
  const [who, setWho] = useState<Who>(null)
  const [history, setHistory] = useState<Creation[]>([])

  const loadHistory = useCallback(() => {
    api.history('generated').then(setHistory, () => {})
  }, [])
  const { job, active, error, start, cancel } = useJob(() => {
    onChange()
    loadHistory()
  })

  useEffect(() => {
    const id = setTimeout(loadHistory, 0)
    return () => clearTimeout(id)
  }, [loadHistory])

  const fill = useCallback((c: Creation) => {
    setPrompt(c.params.prompt ?? '')
    setAvoid(c.params.negative_prompt ?? '')
    setSize(c.params.size ?? 'square')
    setStyle(c.params.style ?? '')
    setSeed(c.params.seed !== undefined ? String(c.params.seed) : '')
    setWho(c.params.person_id ?? null)
    setCount(1)
  }, [])
  useEffect(() => {
    if (!reuse) return
    const id = setTimeout(() => fill(reuse), 0)
    return () => clearTimeout(id)
  }, [reuse, fill])

  // Look for people's names a moment after typing stops.
  useEffect(() => {
    let current = true
    const id = setTimeout(() => {
      if (!prompt.trim()) return setMentions([])
      api.promptPeople(prompt).then(
        (r) => current && setMentions(r.mentions),
        () => {},
      )
    }, 350)
    return () => {
      current = false
      clearTimeout(id)
    }
  }, [prompt])

  const named = mentions.filter((m) => m.status !== 'unknown')
  const unknown = mentions.filter((m) => m.status === 'unknown')
  const candidates = useMemo(() => {
    const seen = new Map<number, PersonCandidate>()
    for (const m of named) for (const c of m.candidates) seen.set(c.person_id, c)
    return [...seen.values()]
  }, [named])
  // One clear name needs no question; anything else waits for the user's choice.
  const obvious = named.length === 1 && named[0].status === 'match' ? named[0].candidates[0].person_id : null
  const chosen: Who =
    who === 'nobody' ? 'nobody' : who !== null && candidates.some((c) => c.person_id === who) ? who : obvious
  const undecided = named.length > 0 && chosen === null
  const personId = typeof chosen === 'number' ? chosen : undefined
  const person = candidates.find((c) => c.person_id === personId)

  if (!status) return <main className="create" />

  const styleModel = style ? [`lora-${style}`] : []
  const needed = [
    ...status.features[personId !== undefined ? 'generate_person' : 'generate'],
    ...styleModel.filter((m) => status.models.some((x) => x.id === m && !x.installed)),
  ]

  const generate = () =>
    void start({
      operation: 'generate',
      prompt,
      negative_prompt: avoid,
      size,
      style: style || undefined,
      count,
      seed: seed.trim() ? Number(seed) : undefined,
      person_id: personId,
    })

  return (
    <main className="create">
      <div className="create-form">
        <h2>Create a picture</h2>
        <label className="field">
          <span>Describe it. You can name someone from your People.</span>
          <textarea
            rows={3}
            value={prompt}
            placeholder="Amit playing guitar on stage"
            onChange={(e) => setPrompt(e.target.value)}
          />
        </label>

        {named.length > 0 && (
          <section className="who-box">
            {named.length === 1 && named[0].status === 'match' ? (
              <h3>This will look like</h3>
            ) : named.length === 1 ? (
              <h3>Which “{named[0].text}” do you mean?</h3>
            ) : (
              <h3>A picture can show one person you know for now. Who should it be?</h3>
            )}
            <div className="who-list">
              {candidates.map((c) => (
                <Face key={c.person_id} person={c} selected={chosen === c.person_id} onPick={() => setWho(c.person_id)} />
              ))}
              <button className="who" aria-pressed={chosen === 'nobody'} onClick={() => setWho('nobody')}>
                <span>Nobody I know: just use the words</span>
              </button>
            </div>
            {person && (
              <p className="muted">
                Pixora studies a few of the clearest photos of {person.name} on this computer. Nothing is sent
                anywhere.
              </p>
            )}
          </section>
        )}
        {unknown.length > 0 && (
          <p className="muted">
            Nobody in your People is called {unknown.map((m) => `“${m.text}”`).join(' or ')}, so the picture will not
            look like anyone you know. Name them in People first if it should.
          </p>
        )}

        <div className="field">
          <span>Shape</span>
          <div className="chips">
            {status.sizes.map((s) => (
              <button key={s.id} aria-pressed={size === s.id} onClick={() => setSize(s.id)}>
                {s.label}
                <span className="muted">
                  {' '}
                  {s.width}×{s.height}
                </span>
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <span>Look</span>
          <div className="chips">
            <button aria-pressed={style === ''} onClick={() => setStyle('')}>
              None
            </button>
            {status.styles.map((s) => (
              <button key={s.id} aria-pressed={style === s.id} onClick={() => setStyle(s.id)}>
                {s.label}
              </button>
            ))}
          </div>
        </div>
        <label className="field">
          <span>Things to avoid (optional)</span>
          <input value={avoid} placeholder="rain, crowds" onChange={(e) => setAvoid(e.target.value)} />
        </label>
        <div className="row">
          <label className="field">
            <span>Versions</span>
            <select value={count} onChange={(e) => setCount(Number(e.target.value))}>
              {Array.from({ length: status.max_count }, (_, i) => i + 1).map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Seed (empty for a new picture each time)</span>
            <input
              inputMode="numeric"
              value={seed}
              placeholder="random"
              onChange={(e) => setSeed(e.target.value.replace(/\D/g, '').slice(0, 9))}
            />
          </label>
        </div>

        {needed.length > 0 ? (
          <ModelsNeeded status={status} needed={needed} onInstalled={() => setInstalled((n) => n + 1)} />
        ) : job && active ? (
          <JobProgress job={job} onCancel={cancel} />
        ) : (
          <button className="primary" disabled={!prompt.trim() || undecided} onClick={generate}>
            Create
          </button>
        )}
        {undecided && <p className="muted">Choose who the picture should show first.</p>}
        {job?.state === 'cancelled' && <p className="muted">Cancelled. Nothing was saved.</p>}
        {error && <p className="form-error">{error}</p>}
        {job?.state === 'done' && (
          <section>
            <h3>Saved to your library</h3>
            <Results photos={job.results} onOpen={onOpen} />
          </section>
        )}
        <p className="muted">New pictures are saved in {status.creations_dir}</p>
      </div>

      <div className="create-history">
        <h3>Earlier pictures</h3>
        {history.length === 0 && <p className="muted">Pictures you create appear here with the settings used.</p>}
        {history.map((c) => (
          <div key={c.photo.id} className="history-item">
            <button className="tile" onClick={() => onOpen(c.photo)} aria-label="Open picture">
              <img src={api.thumbUrl(c.photo.hash, 256)} alt="" />
            </button>
            <div>
              <div className="history-prompt">{c.params.prompt}</div>
              <div className="muted">
                {[
                  c.params.person_name,
                  status.styles.find((s) => s.id === c.params.style)?.label,
                  status.sizes.find((s) => s.id === c.params.size)?.label,
                  `seed ${c.params.seed}`,
                ]
                  .filter(Boolean)
                  .join(' · ')}
              </div>
              <button className="link" onClick={() => fill(c)}>
                Use these settings
              </button>
            </div>
          </div>
        ))}
      </div>
    </main>
  )
}
