import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { api, type Filters, type GalaxyData, type PhotoSummary } from '../api'
import { buildLayout, type Mode } from '../galaxy/layout'
import { Stage } from '../galaxy/stage'

interface Props {
  filters: Filters
  onShowPhoto: (photo: PhotoSummary) => void
}

const MODES: { mode: Mode; label: string }[] = [
  { mode: 'people', label: 'People' },
  { mode: 'scenes', label: 'Scenes' },
  { mode: 'timeline', label: 'Timeline' },
]
const LABEL_ROW = 30
const UNNAMED = new THREE.Color('#7d8798')
const LINK_FAINT = new THREE.Color('#1b1f26')
const LINK_STRONG = new THREE.Color('#9db8ff')

/** One colour per person, spaced so neighbours in the list differ clearly. */
function personColours(count: number): THREE.Color[] {
  return Array.from({ length: count }, (_, i) => new THREE.Color().setHSL((0.02 + i * 0.618034) % 1, 0.7, 0.62))
}

export function Galaxy({ filters, onShowPhoto }: Props) {
  const mount = useRef<HTMLDivElement>(null)
  const stage = useRef<Stage | null>(null)
  const labels = useRef(new Map<string, HTMLButtonElement>())
  const hoverBox = useRef<HTMLDivElement>(null)
  const [mode, setMode] = useState<Mode>('people')
  const [data, setData] = useState<GalaxyData | null>(null)
  const [failed, setFailed] = useState(false)
  const [hovered, setHovered] = useState<number | null>(null)

  useEffect(() => {
    let current = true
    api.galaxyData(filters).then(
      (d) => {
        if (!current) return
        setData(d)
        setFailed(false)
      },
      () => current && setFailed(true),
    )
    return () => {
      current = false
    }
  }, [filters])

  const layout = useMemo(() => (data ? buildLayout(data, mode) : null), [data, mode])
  const colours = useMemo(() => personColours(data?.people.length ?? 0), [data])

  // The frame callback reads the latest values without restarting the scene.
  const live = useRef({ data, layout, hovered, onShowPhoto })
  useEffect(() => {
    live.current = { data, layout, hovered, onShowPhoto }
  })

  useEffect(() => {
    const s = new Stage(mount.current!, {
      onHover: setHovered,
      onPick: (index) => {
        const face = live.current.data?.faces[index]
        if (!face) return
        live.current.onShowPhoto({
          id: face.photo_id,
          hash: face.hash,
          taken_at: face.taken_at,
          width: face.width,
          height: face.height,
        })
      },
      onFrame: () => {
        const { data, layout, hovered } = live.current
        if (!data || !layout) return
        // Labels sit on the outer side of their group, away from the middle of the scene.
        const middle = s.project(0, 0, 0)
        const mx = middle.visible ? middle.x : 0
        const my = middle.visible ? middle.y : 0
        const shown: { el: HTMLButtonElement; x: number; y: number; half: number; count: number }[] = []
        for (const group of layout.groups) {
          const el = labels.current.get(group.key)
          if (!el) continue
          const p = s.project(...group.centre)
          const radius = group.radius * p.scale
          // Small, distant people stay unlabelled until the view moves closer.
          const onScreen =
            p.visible && p.x > -radius && p.x < s.width + radius && p.y > -radius && p.y < s.height + radius
          const show = onScreen && (group.person === null || radius >= 10)
          el.style.display = show ? '' : 'none'
          if (!show) continue
          let ux = p.x - mx
          let uy = p.y - my
          const away = Math.hypot(ux, uy)
          if (layout.labels === 'above' || !middle.visible || away < 12) {
            ux = 0
            uy = -1
          } else {
            ux /= away
            uy /= away
          }
          const half = el.offsetWidth / 2
          const reach = radius + 6 + Math.abs(ux) * half + (Math.abs(uy) * LABEL_ROW) / 2
          // Kept on screen while its group is, even when the outer side is past the edge.
          const x = Math.min(Math.max(p.x + ux * reach, half + 8), s.width - half - 8)
          const y = Math.min(Math.max(p.y + uy * reach, LABEL_ROW / 2 + 4), s.height - LABEL_ROW / 2 - 4)
          shown.push({ el, x, y, half: half + 3, count: group.count })
        }
        // Largest groups first. A label that would cover another steps up or down to a free row,
        // or stays hidden until the view moves closer.
        shown.sort((a, b) => b.count - a.count)
        const placed: { x: number; y: number; half: number }[] = []
        for (const label of shown) {
          let y: number | null = null
          for (const step of [0, -1, 1]) {
            const row = label.y + step * LABEL_ROW
            if (placed.every((o) => Math.abs(label.x - o.x) > label.half + o.half || Math.abs(row - o.y) >= LABEL_ROW)) {
              y = row
              break
            }
          }
          if (y === null) {
            label.el.style.display = 'none'
            continue
          }
          placed.push({ x: label.x, y, half: label.half })
          label.el.style.transform = `translate(${label.x}px, ${y}px) translate(-50%, -50%)`
        }
        const box = hoverBox.current
        if (box && hovered !== null && hovered * 3 < s.current.length) {
          const p = s.project(s.current[hovered * 3], s.current[hovered * 3 + 1], s.current[hovered * 3 + 2])
          const size = Math.max(data.point_size * p.scale, 10) + 6
          box.style.transform = `translate(${p.x}px, ${p.y}px)`
          box.style.setProperty('--ring', `${size}px`)
        }
      },
    })
    stage.current = s
    return () => {
      s.dispose()
      stage.current = null
    }
  }, [])

  const shownMode = useRef<Mode | null>(null)
  useEffect(() => {
    if (!data || !layout || !stage.current) return
    const index = new Map(data.people.map((p, i) => [p.id, i]))
    const colors = new Float32Array(data.faces.length * 3)
    data.faces.forEach((f, i) => {
      const c = f.person_id === null ? UNNAMED : (colours[index.get(f.person_id)!] ?? UNNAMED)
      colors.set([c.r, c.g, c.b], i * 3)
    })

    // Lines between people, brighter the more photos they share.
    const strongest = Math.max(1, ...data.links.map((l) => l.photos))
    const links = mode === 'people' ? data.links.filter((l) => l.photos / strongest >= 0.25) : []
    const ends = new Float32Array(links.length * 6)
    const endColors = new Float32Array(links.length * 6)
    const shade = new THREE.Color()
    links.forEach((link, i) => {
      const a = data.people[index.get(link.a)!]
      const b = data.people[index.get(link.b)!]
      ends.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6)
      shade.lerpColors(LINK_FAINT, LINK_STRONG, link.photos / strongest)
      endColors.set([shade.r, shade.g, shade.b, shade.r, shade.g, shade.b], i * 6)
    })

    stage.current.setScene({
      ids: data.faces.map((f) => f.id),
      positions: layout.positions,
      colors,
      faint: data.faces.map((f) => f.person_id === null),
      pointSize: data.point_size,
      links: ends,
      linkColors: endColors,
      fitRadius: layout.fitRadius,
      faceUrl: api.faceUrl,
      front: mode !== 'people' || shownMode.current !== 'people',
    })
    shownMode.current = mode
  }, [data, layout, colours, mode])

  const face = hovered !== null ? data?.faces[hovered] : undefined
  const person = face && face.person_id !== null ? data!.people.find((p) => p.id === face.person_id) : undefined
  const named = data?.faces.filter((f) => f.person_id !== null).length ?? 0

  return (
    <main className="galaxy-container">
      <div className="galaxy-main" ref={mount} />

      <div className="galaxy-overlay">
        {layout?.groups.map((group) => (
          <button
            key={group.key}
            ref={(el) => {
              if (el) labels.current.set(group.key, el)
              else labels.current.delete(group.key)
            }}
            className="galaxy-label"
            style={group.person === null ? undefined : { borderColor: `#${colours[group.person]?.getHexString()}` }}
            title={`Zoom to ${group.label}`}
            onClick={() => stage.current?.focus(group.centre, group.radius)}
          >
            {group.person !== null && data?.people[group.person]?.cover_face_id != null && (
              <img src={api.faceUrl(data.people[group.person].cover_face_id!)} alt="" />
            )}
            {group.label}
            <span className="muted">{group.count.toLocaleString()}</span>
          </button>
        ))}

        {face && (
          <div className="galaxy-hover" ref={hoverBox}>
            <div className="galaxy-ring" />
            <div className="galaxy-card">
              <img src={api.faceUrl(face.id)} alt="" />
              <div>
                <strong>{person?.name ?? 'Unnamed'}</strong>
                <div className="muted">
                  {new Date(face.taken_at).toLocaleDateString(undefined, { dateStyle: 'medium' })}
                  {face.scene ? ` · ${face.scene}` : ''}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {failed && !data ? (
        <div className="galaxy-message">Could not load the galaxy. Check that the local service is running.</div>
      ) : !data ? (
        <div className="galaxy-message">Loading galaxy…</div>
      ) : data.faces.length === 0 ? (
        <div className="galaxy-message">
          {Object.keys(filters).length > 0 ? 'No faces in the photos that match these filters.' : 'No faces found yet.'}
        </div>
      ) : null}

      <div className="galaxy-controls">
        <div className="galaxy-modes">
          {MODES.map((m) => (
            <button
              key={m.mode}
              className={mode === m.mode ? 'active' : ''}
              onClick={() => setMode(m.mode)}
              aria-pressed={mode === m.mode}
            >
              {m.label}
            </button>
          ))}
          <button onClick={() => stage.current?.fit(mode !== 'people')}>Reset view</button>
        </div>
      </div>

      <div className="galaxy-info">
        <p>Drag to rotate · Scroll to zoom · Click a face to open its photo · Click a name to zoom in</p>
        {data && data.faces.length > 0 && (
          <p>
            {data.faces.length.toLocaleString()} faces · {data.people.length.toLocaleString()} people
            {data.faces.length > named && ` · ${(data.faces.length - named).toLocaleString()} unnamed faces in grey`}
          </p>
        )}
      </div>
    </main>
  )
}
