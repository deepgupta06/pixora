import { useCallback, useMemo, useState } from 'react'
import type { Creation, Filters, PhotoSummary } from './api'
import { Create } from './components/Create'
import { FilterBar } from './components/FilterBar'
import { AddFolderForm, FoldersDialog } from './components/Folders'
import { Galaxy } from './components/Galaxy'
import { Grid } from './components/Grid'
import { Map } from './components/Map'
import { PeopleView } from './components/People'
import { SyncStatus } from './components/SyncStatus'
import { Viewer } from './components/Viewer'
import { useLibrary } from './useLibrary'

export default function App() {
  // One filter state for every view (the map and galaxy will read it too).
  const [filters, setFilters] = useState<Filters>({})
  const { status, timeline, months, offline, loadMonth, poll } = useLibrary(filters)
  const [foldersOpen, setFoldersOpen] = useState(false)
  const [openId, setOpenId] = useState<number | null>(null)
  const [view, setView] = useState<'photos' | 'map' | 'galaxy' | 'people' | 'create'>('photos')
  // A generated picture whose settings the Create view should start from.
  const [reuse, setReuse] = useState<Creation | null>(null)

  // Viewer order follows the grid: newest month first, across the loaded months.
  const ordered = useMemo(
    () => timeline.buckets.flatMap((b) => months[b.month] ?? []),
    [timeline, months],
  )
  const openIndex = openId === null ? -1 : ordered.findIndex((p) => p.id === openId)
  // A photo opened from the map or galaxy may not be in a month the grid has loaded.
  const [opened, setOpened] = useState<PhotoSummary | null>(null)
  const openPhoto = openIndex >= 0 ? ordered[openIndex] : opened?.id === openId ? opened : null

  const refresh = useCallback(() => void poll(true), [poll])
  const onOpen = useCallback((p: PhotoSummary) => {
    setOpened(p)
    setOpenId(p.id)
  }, [])
  const close = useCallback(() => setOpenId(null), [])
  const prev = useMemo(
    () => (openIndex > 0 ? () => setOpenId(ordered[openIndex - 1].id) : undefined),
    [ordered, openIndex],
  )
  const next = useMemo(
    () =>
      openIndex >= 0 && openIndex < ordered.length - 1
        ? () => setOpenId(ordered[openIndex + 1].id)
        : undefined,
    [ordered, openIndex],
  )

  const empty = status !== null && timeline.total === 0
  const filtered = Object.keys(filters).length > 0
  const hasPhotos = (status?.library.photos ?? 0) > 0
  const working = status?.state === 'discovering' || status?.state === 'processing'

  return (
    <div className="app">
      <header className="topbar">
        <h1>Pixora</h1>
        <nav className="tabs" aria-label="Views">
          {(['photos', 'map', 'galaxy', 'people', 'create'] as const).map((v) => (
            <button key={v} aria-current={view === v ? 'page' : undefined} onClick={() => setView(v)}>
              {{ photos: 'Photos', map: 'Map', galaxy: 'Galaxy', people: 'People', create: 'Create' }[v]}
            </button>
          ))}
        </nav>
        <SyncStatus status={status} offline={offline} onChange={refresh} />
        <button onClick={() => setFoldersOpen(true)}>Folders</button>
      </header>

      {view === 'people' ? (
        <PeopleView
          facesDone={status?.stages.faces.done ?? 0}
          note={status?.faces_note ?? null}
          onShowPerson={(personId) => {
            setFilters({ personId })
            setView('photos')
          }}
        />
      ) : view === 'create' ? (
        <Create onOpen={onOpen} onChange={refresh} reuse={reuse} />
      ) : view === 'map' ? (
        <Map filters={filters} onShowPhoto={onOpen} />
      ) : view === 'galaxy' ? (
        <Galaxy filters={filters} onShowPhoto={onOpen} />
      ) : (
        <>
      {hasPhotos && (
        <FilterBar
          filters={filters}
          onChange={setFilters}
          matching={timeline.total}
          libraryPhotos={status?.library.photos ?? 0}
        />
      )}

      {offline && !status ? (
        <main className="empty">
          <h2>Local service is not running</h2>
          <p className="muted">
            Start it from the <code>service</code> folder with <code>python -m pixora_service</code>.
          </p>
        </main>
      ) : empty && status.library.files === 0 && !working ? (
        <main className="empty">
          <h2>Add a folder to get started</h2>
          <p className="muted">
            Pixora scans it in the background and shows photos as they are ready. Your photos stay on this
            computer and are never changed.
          </p>
          <AddFolderForm onAdded={refresh} />
        </main>
      ) : empty && filtered ? (
        <main className="empty">
          <h2>No photos match these filters</h2>
          <button onClick={() => setFilters({})}>Clear filters</button>
        </main>
      ) : empty ? (
        <main className="empty">
          <h2>{working ? 'Looking for photos…' : 'No photos to show yet'}</h2>
          <p className="muted">
            {working
              ? 'They will appear here as soon as their thumbnails are ready.'
              : 'Resume the sync or add another folder.'}
          </p>
        </main>
      ) : (
        <Grid key={JSON.stringify(filters)} timeline={timeline} months={months} loadMonth={loadMonth} onOpen={onOpen} />
      )}
        </>
      )}

      <FoldersDialog open={foldersOpen} onClose={() => setFoldersOpen(false)} onChange={refresh} />
      {openPhoto && (
        <Viewer
          photo={openPhoto}
          onClose={close}
          onPrev={prev}
          onNext={next}
          onOpen={onOpen}
          onChange={refresh}
          onReuse={(creation) => {
            setReuse(creation)
            setView('create')
            close()
          }}
        />
      )}
    </div>
  )
}
