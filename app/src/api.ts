// Typed client for the local Pixora service. These types mirror the pydantic
// models in service/pixora_service/api.py.

export interface Folder {
  id: number
  path: string
  added_at: string
  file_count: number
  error: string | null
}

export interface StageProgress {
  done: number
  total: number
}

export type ScanState = 'idle' | 'discovering' | 'processing' | 'faces' | 'scenes' | 'paused'

export interface ScanStatus {
  state: ScanState
  paused: boolean
  discovered: number
  stages: { hash: StageProgress; thumbnail: StageProgress; faces: StageProgress; scenes: StageProgress }
  /** Why the face stage is not running, when it is not. */
  faces_note: string | null
  scenes_note: string | null
  errors: number
  library: { photos: number; files: number; duplicate_files: number }
  folder_errors: { folder_id: number; error: string }[]
}

export interface Timeline {
  total: number
  buckets: { month: string; count: number }[]
}

export interface Facets {
  cameras: { camera: string; count: number }[]
  people: { id: number; name: string; count: number }[]
  scenes: { name: string; category: string; count: number }[]
  occasions: { name: string; count: number }[]
  /** False until photos have been prepared for search. */
  search_available: boolean
  first_date: string | null
  last_date: string | null
}

export interface MapPhoto {
  id: number
  lat: number
  lon: number
  hash: string
}

export interface MapCluster {
  id: number
  lat: number
  lon: number
  count: number
  expansion_zoom: number
}

export interface MapData {
  photos_with_location: number
  photos_without_location: number
  points: (MapPhoto | MapCluster)[]
}

export interface NoLocationPhotos {
  photos: PhotoSummary[]
}

export interface LocationSuggestions {
  photo_id: number
  suggestions: { lat: number; lon: number; place: string }[]
}

export interface GalaxyFace {
  id: number
  photo_id: number
  hash: string
  taken_at: string
  width: number
  height: number
  person_id: number | null
  /** The photo's strongest scene tag, if any. */
  scene: string | null
  /** Position in People mode. */
  x: number
  y: number
  z: number
}

export interface GalaxyPerson {
  id: number
  name: string
  face_count: number
  cover_face_id: number | null
  x: number
  y: number
  z: number
  radius: number
}

export interface GalaxyData {
  faces: GalaxyFace[]
  people: GalaxyPerson[]
  /** Pairs of people and the number of photos that show them together. */
  links: { a: number; b: number; photos: number }[]
  /** Size of one point, in the same units as the positions. */
  point_size: number
}

/** The filter state shared by every view. Dates are YYYY-MM-DD, inclusive. */
export interface Filters {
  /** A description in plain words, or a person's name. */
  q?: string
  scene?: string
  occasion?: string
  personId?: number
  camera?: string
  dateFrom?: string
  dateTo?: string
}

function filterQuery(f: Filters): string {
  const q = new URLSearchParams()
  if (f.q) q.set('q', f.q)
  if (f.scene) q.set('scene', f.scene)
  if (f.occasion) q.set('occasion', f.occasion)
  if (f.personId) q.set('person_id', String(f.personId))
  if (f.camera) q.set('camera', f.camera)
  if (f.dateFrom) q.set('date_from', f.dateFrom)
  if (f.dateTo) q.set('date_to', f.dateTo)
  return q.toString()
}

export interface PhotoSummary {
  id: number
  hash: string
  taken_at: string
  width: number
  height: number
}

export interface PhotoDetail extends PhotoSummary {
  taken_at_source: 'exif' | 'file'
  lat: number | null
  lon: number | null
  camera: string | null
  files: { path: string; size: number; missing: boolean }[]
  faces: PhotoFace[]
  tags: { name: string; category: string; score: number }[]
  event: PhotoEvent | null
}

export interface PhotoEvent {
  id: number
  /** The place most of the event's photos show, or empty. */
  title: string
  /** The inferred occasion, if any. */
  type: string | null
  start_at: string
  end_at: string
  photo_count: number
}

export interface PhotoFace {
  id: number
  /** Box as fractions of the photo's width and height. */
  x1: number
  y1: number
  x2: number
  y2: number
  person_id: number | null
  name: string | null
  source: 'user' | 'auto' | null
  suggested_person_id: number | null
  suggested_name: string | null
}

export interface Person {
  id: number
  name: string
  hidden: boolean
  cover_face_id: number | null
  face_count: number
  photo_count: number
}

export interface PeopleOverview {
  people: Person[]
  unnamed_groups: number
}

/** "Who is this?" (name), "Is this also NAME?" (confirm), or nothing left to ask. */
export interface Question {
  type: 'name' | 'confirm' | 'done'
  face_ids: number[]
  sample_face_ids: number[]
  photo_count: number
  person_id: number | null
  name: string | null
}

/** True inside the Tauri desktop shell. */
export const isDesktop = '__TAURI_INTERNALS__' in window

// In development Vite proxies /api, so requests stay same-origin. The packaged
// desktop app is served from its own origin and calls the service directly.
const BASE: string =
  import.meta.env.VITE_API_BASE ?? (isDesktop && !import.meta.env.DEV ? 'http://127.0.0.1:8732' : '')

async function request<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  const res = await fetch(BASE + path, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const data = await res.json()
      if (typeof data.detail === 'string') detail = data.detail
    } catch {
      // keep the status text
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  folders: () => request<Folder[]>('/api/folders'),
  addFolder: (path: string) => request<Folder[]>('/api/folders', 'POST', { path }),
  removeFolder: (id: number) => request<Folder[]>(`/api/folders/${id}`, 'DELETE'),
  status: () => request<ScanStatus>('/api/scan/status'),
  startScan: () => request<ScanStatus>('/api/scan/start', 'POST'),
  pauseScan: () => request<ScanStatus>('/api/scan/pause', 'POST'),
  resumeScan: () => request<ScanStatus>('/api/scan/resume', 'POST'),
  facets: () => request<Facets>('/api/facets'),
  timeline: (f: Filters) => request<Timeline>(`/api/timeline?${filterQuery(f)}`),
  photos: (month: string, f: Filters) =>
    request<PhotoSummary[]>(`/api/photos?month=${month}&${filterQuery(f)}`),
  photo: (id: number) => request<PhotoDetail>(`/api/photos/${id}`),
  people: () => request<PeopleOverview>('/api/people'),
  question: () => request<Question>('/api/people/question'),
  nameFaces: (faceIds: number[], name: string) =>
    request<{ person_id: number }>('/api/people/name', 'POST', { face_ids: faceIds, name }),
  confirmFaces: (personId: number, faceIds: number[], answer: boolean) =>
    request<Question>('/api/people/confirm', 'POST', { person_id: personId, face_ids: faceIds, answer }),
  dismissFaces: (faceIds: number[]) => request<Question>('/api/people/dismiss', 'POST', { face_ids: faceIds }),
  updatePerson: (id: number, change: { name?: string; hidden?: boolean }) =>
    request<{ person_id: number }>(`/api/people/${id}`, 'PATCH', change),
  assignFace: (faceId: number, name: string) =>
    request<{ person_id: number }>(`/api/faces/${faceId}/assign`, 'POST', { name }),
  rejectFace: (faceId: number) => request<{ ok: boolean }>(`/api/faces/${faceId}/reject`, 'POST'),
  mapData: (f: Filters, zoom: number, bbox: { min_lat: number; max_lat: number; min_lon: number; max_lon: number }) =>
    request<MapData>(
      `/api/map/data?${filterQuery(f)}&zoom=${zoom}&min_lat=${bbox.min_lat}&max_lat=${bbox.max_lat}&min_lon=${bbox.min_lon}&max_lon=${bbox.max_lon}`,
    ),
  noLocationPhotos: (f: Filters) => request<NoLocationPhotos>(`/api/map/no-location?${filterQuery(f)}`),
  locationSuggestions: (photoId: number) => request<LocationSuggestions>(`/api/map/suggestions/${photoId}`),
  galaxyData: (f: Filters) => request<GalaxyData>(`/api/galaxy/data?${filterQuery(f)}`),
  faceUrl: (faceId: number) => `${BASE}/api/faces/${faceId}/crop`,
  thumbUrl: (hash: string, size: 256 | 1024) => `${BASE}/api/thumbs/${hash}/${size}`,
  originalUrl: (id: number) => `${BASE}/api/photos/${id}/original`,
}
