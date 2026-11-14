import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  type PointerEvent,
  type RefObject,
} from 'react'
import { type PhotoSummary, type StudioJob, type StudioJobRequest, type StudioStatus } from '../api'
import { useJob, useStudioStatus } from '../useStudio'
import { JobProgress, ModelsNeeded, Results } from './StudioParts'

export type EditTool = 'background' | 'quality' | 'style' | 'brush' | 'extend'

const TOOLS: { id: EditTool; label: string }[] = [
  { id: 'background', label: 'Background' },
  { id: 'quality', label: 'Quality' },
  { id: 'style', label: 'Style' },
  { id: 'brush', label: 'Brush' },
  { id: 'extend', label: 'Extend' },
]

export interface BrushHandle {
  /** The painted area as a PNG data URL, or null when nothing is painted. */
  mask: () => string | null
  clear: () => void
}

/** A transparent layer over the displayed photo to paint a selection on. */
export const BrushCanvas = forwardRef<BrushHandle, { image: RefObject<HTMLImageElement | null>; size: number }>(
  function BrushCanvas({ image, size }, ref) {
    const canvas = useRef<HTMLCanvasElement>(null)
    const [box, setBox] = useState({ left: 0, top: 0, width: 0, height: 0 })
    const last = useRef<{ x: number; y: number } | null>(null)
    const painted = useRef(false)

    // Keep the layer exactly over the photo as the window changes size.
    useEffect(() => {
      const img = image.current
      if (!img) return
      const measure = () =>
        setBox({ left: img.offsetLeft, top: img.offsetTop, width: img.offsetWidth, height: img.offsetHeight })
      measure()
      const observer = new ResizeObserver(measure)
      observer.observe(img)
      img.addEventListener('load', measure)
      return () => {
        observer.disconnect()
        img.removeEventListener('load', measure)
      }
    }, [image])

    useImperativeHandle(ref, () => ({
      mask: () => (painted.current && canvas.current ? canvas.current.toDataURL('image/png') : null),
      clear: () => {
        const c = canvas.current
        c?.getContext('2d')?.clearRect(0, 0, c.width, c.height)
        painted.current = false
      },
    }))

    const draw = (e: PointerEvent<HTMLCanvasElement>) => {
      const c = canvas.current
      const ctx = c?.getContext('2d')
      if (!c || !ctx) return
      const rect = c.getBoundingClientRect()
      const point = { x: e.clientX - rect.left, y: e.clientY - rect.top }
      ctx.strokeStyle = ctx.fillStyle = '#7aa2ff'
      ctx.lineWidth = size
      ctx.lineCap = ctx.lineJoin = 'round'
      ctx.beginPath()
      ctx.moveTo((last.current ?? point).x, (last.current ?? point).y)
      ctx.lineTo(point.x, point.y)
      ctx.stroke()
      ctx.beginPath()
      ctx.arc(point.x, point.y, size / 2, 0, Math.PI * 2)
      ctx.fill()
      last.current = point
      painted.current = true
    }

    return (
      <canvas
        ref={canvas}
        className="brush-canvas"
        // Keyed by size so a resized window starts a clean layer of the right shape.
        key={`${box.width}x${box.height}`}
        width={box.width}
        height={box.height}
        style={{ left: box.left, top: box.top, width: box.width, height: box.height }}
        aria-label="Paint over the area to change"
        onClick={(e) => e.stopPropagation()}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          last.current = null
          draw(e)
        }}
        onPointerMove={(e) => e.buttons === 1 && draw(e)}
        onPointerUp={() => (last.current = null)}
      />
    )
  },
)

interface Props {
  photo: PhotoSummary
  tool: EditTool
  onTool: (tool: EditTool) => void
  brush: RefObject<BrushHandle | null>
  brushSize: number
  onBrushSize: (size: number) => void
  /** A finished result to show in the viewer. */
  onResult: (photo: PhotoSummary) => void
  /** The library gained pictures. */
  onChange: () => void
}

const EXTEND_STEPS = [0, 0.25, 0.5, 1]
const SIDES = ['left', 'right', 'top', 'bottom'] as const

function Variations({ value, max, onChange }: { value: number; max: number; onChange: (n: number) => void }) {
  return (
    <label className="field">
      <span>Versions to make</span>
      <select value={value} onChange={(e) => onChange(Number(e.target.value))}>
        {Array.from({ length: max }, (_, i) => i + 1).map((n) => (
          <option key={n} value={n}>
            {n}
          </option>
        ))}
      </select>
    </label>
  )
}

export function EditPanel({ photo, tool, onTool, brush, brushSize, onBrushSize, onResult, onChange }: Props) {
  const [installed, setInstalled] = useState(0)
  const status = useStudioStatus(installed)
  const { job, active, error, start, cancel, reset } = useJob((done: StudioJob) => {
    onChange()
    // A single result is shown at once; several are offered to choose from.
    if (done.results.length === 1) onResult(done.results[0])
  })

  const [background, setBackground] = useState('')
  const [faces, setFaces] = useState(false)
  const [style, setStyle] = useState('anime')
  // Null until the slider is moved: each style then uses its own default.
  const [chosenStrength, setStrength] = useState<number | null>(null)
  const [describe, setDescribe] = useState('')
  const [count, setCount] = useState(1)
  const [replacement, setReplacement] = useState('')
  const [extend, setExtend] = useState({ left: 0.25, right: 0.25, top: 0, bottom: 0 })
  const [extendPrompt, setExtendPrompt] = useState('')
  const [unpainted, setUnpainted] = useState(false)

  // A different photo starts clean.
  useEffect(() => reset(), [photo.id, reset])

  const run = (request: Omit<StudioJobRequest, 'photo_id'>) => void start({ ...request, photo_id: photo.id })
  const missing = (feature: string, extra: string[] = []) =>
    status ? [...status.features[feature], ...extra.filter((m) => !status.models.find((x) => x.id === m)?.installed)] : []
  const gate = (status: StudioStatus, needed: string[]) =>
    needed.length > 0 ? (
      <ModelsNeeded status={status} needed={needed} onInstalled={() => setInstalled((n) => n + 1)} />
    ) : null

  const styles = status?.styles.filter((s) => s.restyle) ?? []
  const strength = chosenStrength ?? styles.find((s) => s.id === style)?.strength ?? 0.6
  const max = status?.max_count ?? 1

  return (
    <div className="edit">
      <div className="edit-tools" role="tablist" aria-label="Editing tools">
        {TOOLS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tool === t.id} disabled={active} onClick={() => onTool(t.id)}>
            {t.label}
          </button>
        ))}
      </div>
      <p className="muted edit-note">Your photo is never changed. Every result is saved as a new picture.</p>

      {!status ? (
        <p className="muted">Checking what is installed…</p>
      ) : tool === 'background' ? (
        <>
          <section>
            <h3>Remove background</h3>
            {gate(status, missing('remove_background')) ?? (
              <button disabled={active} onClick={() => run({ operation: 'remove_background' })}>
                Remove background
              </button>
            )}
          </section>
          <section>
            <h3>Change background</h3>
            {gate(status, missing('replace_background')) ?? (
              <>
                <label className="field">
                  <span>New background</span>
                  <textarea
                    rows={2}
                    value={background}
                    placeholder="a beach at sunset"
                    onChange={(e) => setBackground(e.target.value)}
                  />
                </label>
                <Variations value={count} max={max} onChange={setCount} />
                <button
                  disabled={active || !background.trim()}
                  onClick={() => run({ operation: 'replace_background', prompt: background, count })}
                >
                  Change background
                </button>
              </>
            )}
          </section>
        </>
      ) : tool === 'quality' ? (
        gate(status, missing('enhance')) ?? (
          <>
            <section>
              <h3>Increase quality</h3>
              <p className="muted">Makes the picture larger and sharper.</p>
              <label className="check">
                <input type="checkbox" checked={faces} onChange={(e) => setFaces(e.target.checked)} />
                Also sharpen faces
              </label>
              <button disabled={active} onClick={() => run({ operation: 'enhance', faces })}>
                Increase quality
              </button>
            </section>
            <section>
              <h3>Restore an old photo</h3>
              <p className="muted">Rebuilds blurry or damaged faces, then sharpens the whole picture.</p>
              <button disabled={active} onClick={() => run({ operation: 'restore' })}>
                Restore
              </button>
            </section>
          </>
        )
      ) : tool === 'style' ? (
        <section>
          <h3>Redraw in a style</h3>
          <div className="chips">
            {styles.map((s) => (
              <button
                key={s.id}
                aria-pressed={style === s.id}
                onClick={() => {
                  setStyle(s.id)
                  setStrength(null)
                }}
              >
                {s.label}
              </button>
            ))}
          </div>
          {gate(status, missing('stylize', [`lora-${style}`])) ?? (
            <>
              <label className="field">
                <span>How far from the photo: {strength < 0.45 ? 'close' : strength < 0.7 ? 'balanced' : 'free'}</span>
                <input
                  type="range"
                  min={0.3}
                  max={0.9}
                  step={0.05}
                  value={strength}
                  onChange={(e) => setStrength(Number(e.target.value))}
                />
              </label>
              <label className="field">
                <span>What the photo shows (optional, helps keep people as they are)</span>
                <input
                  value={describe}
                  placeholder="two young men at a restaurant"
                  onChange={(e) => setDescribe(e.target.value)}
                />
              </label>
              <Variations value={count} max={max} onChange={setCount} />
              <button
                disabled={active}
                onClick={() => run({ operation: 'stylize', style, strength, prompt: describe, count })}
              >
                Apply style
              </button>
            </>
          )}
        </section>
      ) : tool === 'brush' ? (
        <section>
          <h3>Remove or replace something</h3>
          {gate(status, missing('inpaint')) ?? (
            <>
              <p className="muted">Paint over it in the photo.</p>
              <label className="field">
                <span>Brush size</span>
                <input
                  type="range"
                  min={8}
                  max={120}
                  value={brushSize}
                  onChange={(e) => onBrushSize(Number(e.target.value))}
                />
              </label>
              <label className="field">
                <span>Replace with (leave empty to remove it)</span>
                <input value={replacement} placeholder="a vase of flowers" onChange={(e) => setReplacement(e.target.value)} />
              </label>
              <Variations value={count} max={max} onChange={setCount} />
              <div className="row">
                <button
                  disabled={active}
                  onClick={() => {
                    const mask = brush.current?.mask()
                    setUnpainted(!mask)
                    if (mask) run({ operation: 'inpaint', mask, prompt: replacement, count })
                  }}
                >
                  {replacement.trim() ? 'Replace' : 'Remove'}
                </button>
                <button disabled={active} onClick={() => brush.current?.clear()}>
                  Clear painting
                </button>
              </div>
              {unpainted && <p className="form-error">Paint over part of the photo first.</p>}
            </>
          )}
        </section>
      ) : (
        <section>
          <h3>Extend the picture</h3>
          {gate(status, missing('extend')) ?? (
            <>
              <p className="muted">Adds new picture around the photo, continuing the scene.</p>
              <div className="extend-grid">
                {SIDES.map((side) => (
                  <label key={side} className="field">
                    <span>{side.charAt(0).toUpperCase() + side.slice(1)}</span>
                    <select
                      value={extend[side]}
                      onChange={(e) => setExtend({ ...extend, [side]: Number(e.target.value) })}
                    >
                      {EXTEND_STEPS.map((v) => (
                        <option key={v} value={v}>
                          {v === 0 ? 'None' : `+${v * 100}%`}
                        </option>
                      ))}
                    </select>
                  </label>
                ))}
              </div>
              <label className="field">
                <span>What to add (optional)</span>
                <input value={extendPrompt} placeholder="more of the garden" onChange={(e) => setExtendPrompt(e.target.value)} />
              </label>
              <Variations value={count} max={max} onChange={setCount} />
              <button
                disabled={active || !SIDES.some((s) => extend[s] > 0)}
                onClick={() => run({ operation: 'extend', ...extend, prompt: extendPrompt, count })}
              >
                Extend
              </button>
            </>
          )}
        </section>
      )}

      {job && active && <JobProgress job={job} onCancel={cancel} />}
      {job?.state === 'cancelled' && <p className="muted">Cancelled. Nothing was saved.</p>}
      {error && <p className="form-error">{error}</p>}
      {job?.state === 'done' && job.results.length > 1 && (
        <section>
          <h3>{job.results.length} versions saved</h3>
          <Results photos={job.results} onOpen={onResult} />
        </section>
      )}
    </div>
  )
}
