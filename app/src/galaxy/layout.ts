import type { GalaxyData, GalaxyFace } from '../api'

export type Mode = 'people' | 'scenes' | 'timeline'

export interface Group {
  key: string
  label: string
  count: number
  centre: [number, number, number]
  radius: number
  /** Set for people: the index into `data.people`. */
  person: number | null
}

export interface Layout {
  /** x, y, z for each face, in the order of `data.faces`. */
  positions: Float32Array
  groups: Group[]
  /** Where a group's label goes: over it, or on its side away from the middle. */
  labels: 'above' | 'outward'
  /** Radius of the sphere, centred on the origin, that holds every point. */
  fitRadius: number
}

const GAP = 1.2

// Mirrors cluster_radius in service/pixora_service/galaxy.py.
function ballRadius(count: number, pointSize: number): number {
  return 0.2 + pointSize * Math.cbrt(count)
}

/** `count` places spread evenly through a ball, ordered from top to bottom. */
function ballSlots(count: number, radius: number): [number, number, number][] {
  if (count === 1) return [[0, 0, 0]]
  const frac = (v: number) => v - Math.floor(v)
  const slots: [number, number, number][] = []
  for (let i = 0; i < count; i++) {
    // An evenly spreading sequence; the constants come from the generalised golden ratio.
    const r = radius * Math.cbrt(frac(0.5 + i * 0.8191725134))
    const y = 1 - 2 * frac(0.5 + i * 0.6710436067)
    const angle = 2 * Math.PI * frac(0.5 + i * 0.5497004779)
    const flat = Math.sqrt(1 - y * y)
    slots.push([r * flat * Math.cos(angle), r * y, r * flat * Math.sin(angle)])
  }
  return slots.sort((a, b) => b[1] - a[1])
}

function extent(positions: Float32Array): number {
  let max = 0
  for (let i = 0; i < positions.length; i += 3) {
    max = Math.max(max, Math.hypot(positions[i], positions[i + 1], positions[i + 2]))
  }
  return max
}

function peopleLayout(data: GalaxyData): Layout {
  const positions = new Float32Array(data.faces.length * 3)
  data.faces.forEach((f, i) => positions.set([f.x, f.y, f.z], i * 3))
  // Frame the named people; unnamed faces surround them as a backdrop.
  const core = Math.max(0, ...data.people.map((p) => Math.hypot(p.x, p.y, p.z) + p.radius))
  return {
    positions,
    groups: data.people.map((p, i) => ({
      key: `person-${p.id}`,
      label: p.name,
      count: p.face_count,
      centre: [p.x, p.y, p.z],
      radius: p.radius,
      person: i,
    })),
    labels: 'outward',
    fitRadius: core > 0 ? core * 1.3 : Math.max(extent(positions), 1),
  }
}

/** The same points gathered into one ball per key, with the balls in a row or a ring. */
function groupedLayout(
  data: GalaxyData,
  keyOf: (face: GalaxyFace) => string,
  labelOf: (key: string) => string,
  shape: 'row' | 'ring',
): Layout {
  const members = new Map<string, number[]>()
  data.faces.forEach((f, i) => {
    const key = keyOf(f)
    const list = members.get(key)
    if (list) list.push(i)
    else members.set(key, [i])
  })
  const keys = [...members.keys()]
  if (shape === 'row') keys.sort()
  // Largest first, with the "none" group last.
  else keys.sort((a, b) => Number(a === '') - Number(b === '') || members.get(b)!.length - members.get(a)!.length)

  const radii = keys.map((k) => ballRadius(members.get(k)!.length, data.point_size))
  const length = radii.reduce((sum, r) => sum + 2 * r + GAP, 0)
  const ring = keys.length > 1 ? Math.max(length / (2 * Math.PI), Math.max(...radii)) : 0

  const positions = new Float32Array(data.faces.length * 3)
  const groups: Group[] = []
  let along = 0
  keys.forEach((key, g) => {
    const mid = along + radii[g] + GAP / 2
    along += 2 * radii[g] + GAP
    const angle = (mid / length) * 2 * Math.PI
    const centre: [number, number, number] =
      shape === 'row' ? [mid - length / 2, 0, 0] : [ring * Math.sin(angle), ring * Math.cos(angle), 0]

    // Each person's faces stay together as a layer of the ball, with unnamed faces at the bottom.
    const faces = members.get(key)!
    faces.sort((a, b) => (data.faces[a].person_id ?? Infinity) - (data.faces[b].person_id ?? Infinity) || a - b)
    const slots = ballSlots(faces.length, radii[g])
    faces.forEach((index, i) => {
      positions[index * 3] = centre[0] + slots[i][0]
      positions[index * 3 + 1] = centre[1] + slots[i][1]
      positions[index * 3 + 2] = centre[2] + slots[i][2]
    })
    groups.push({ key: `${shape}-${key}`, label: labelOf(key), count: faces.length, centre, radius: radii[g], person: null })
  })
  return {
    positions,
    groups,
    labels: shape === 'row' ? 'above' : 'outward',
    fitRadius: Math.max(extent(positions), 1) * 1.15,
  }
}

export function buildLayout(data: GalaxyData, mode: Mode): Layout {
  if (mode === 'timeline') {
    return groupedLayout(data, (f) => f.taken_at.slice(0, 4), (year) => year, 'row')
  }
  if (mode === 'scenes') {
    return groupedLayout(
      data,
      (f) => f.scene ?? '',
      (scene) => (scene ? scene.charAt(0).toUpperCase() + scene.slice(1) : 'No scene'),
      'ring',
    )
  }
  return peopleLayout(data)
}
