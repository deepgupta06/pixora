import { useRef, useState, type PointerEvent } from 'react'
import { api, type PhotoSummary, type StudioJob, type StudioStatus } from '../api'
import { useJob } from '../useStudio'

/** Progress of a running job, with a way to stop it. */
export function JobProgress({ job, onCancel }: { job: StudioJob; onCancel: () => void }) {
  const percent = Math.round(job.progress * 100)
  return (
    <div className="job" role="status">
      <div className="job-line">
        <span>
          {job.state === 'queued' ? 'Waiting for another job to finish' : job.message}
          {job.state === 'running' && percent > 0 && ` · ${percent}%`}
        </span>
        <button onClick={onCancel}>Cancel</button>
      </div>
      <progress max={1} value={job.state === 'queued' ? undefined : job.progress} aria-label="Progress" />
    </div>
  )
}

/** "Edited" or "Generated", for pictures Pixora made. */
export function OriginBadge({ origin }: { origin: PhotoSummary['origin'] }) {
  if (!origin) return null
  return <span className={`badge badge-${origin}`}>{origin === 'edited' ? 'Edited' : 'Generated'}</span>
}

/** A row of result thumbnails that open in the viewer. */
export function Results({ photos, onOpen }: { photos: PhotoSummary[]; onOpen: (p: PhotoSummary) => void }) {
  return (
    <div className="results">
      {photos.map((p) => (
        <button key={p.id} className="tile result" onClick={() => onOpen(p)} aria-label="Open result">
          <img src={api.thumbUrl(p.hash, 256)} alt="" />
        </button>
      ))}
    </div>
  )
}

function megabytes(mb: number): string {
  return mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`
}

/**
 * Shown in place of a feature whose models are not installed: what is needed,
 * under which licence, and a button that downloads it.
 */
export function ModelsNeeded({
  status,
  needed,
  onInstalled,
}: {
  status: StudioStatus
  needed: string[]
  onInstalled: () => void
}) {
  const { job, active, error, start, cancel } = useJob(onInstalled)
  const models = status.models.filter((m) => needed.includes(m.id))
  const total = models.reduce((sum, m) => sum + m.size_mb, 0)
  return (
    <div className="models-needed">
      <p>
        This needs {models.length === 1 ? 'a model that is' : 'models that are'} not on this computer yet (
        {megabytes(total)}). They are downloaded once and then run here; your photos are never sent anywhere.
      </p>
      <ul>
        {models.map((m) => (
          <li key={m.id}>
            {m.label}
            <span className="muted">
              {' '}
              · {megabytes(m.size_mb)} · {m.licence}
            </span>
          </li>
        ))}
      </ul>
      {!status.engine_installed && (
        <p className="form-error">
          The editing engine (PyTorch) is not installed. In the <code>service</code> folder run{' '}
          <code>.venv\Scripts\python -m pip install -e ".[studio]" --extra-index-url https://download.pytorch.org/whl/cu128</code>
        </p>
      )}
      {job && active ? (
        <JobProgress job={job} onCancel={cancel} />
      ) : (
        <button className="primary" onClick={() => void start({ operation: 'download', models: needed })}>
          Download {megabytes(total)}
        </button>
      )}
      {error && <p className="form-error">{error}</p>}
    </div>
  )
}

/** Two pictures of the same shape with a divider to drag between them. */
export function Compare({ before, after }: { before: string; after: string }) {
  const [split, setSplit] = useState(0.5)
  const ref = useRef<HTMLDivElement>(null)
  const move = (e: PointerEvent) => {
    if (e.buttons === 0 && e.type !== 'pointerdown') return
    const box = ref.current?.getBoundingClientRect()
    if (box) setSplit(Math.min(1, Math.max(0, (e.clientX - box.left) / box.width)))
  }
  return (
    <div
      ref={ref}
      className="compare"
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId)
        move(e)
      }}
      onPointerMove={move}
      onClick={(e) => e.stopPropagation()}
    >
      <img src={after} alt="Result" draggable={false} />
      <img
        className="compare-before"
        src={before}
        alt="Original"
        draggable={false}
        style={{ clipPath: `inset(0 ${(1 - split) * 100}% 0 0)` }}
      />
      <div className="compare-line" style={{ left: `${split * 100}%` }}>
        <span>‹ ›</span>
      </div>
      <span className="compare-label compare-label-left">Original</span>
      <span className="compare-label compare-label-right">Result</span>
      <input
        className="compare-range"
        type="range"
        min={0}
        max={100}
        value={Math.round(split * 100)}
        onChange={(e) => setSplit(Number(e.target.value) / 100)}
        aria-label="How much of the original to show"
      />
    </div>
  )
}
