import { useEffect, useState, type FormEvent } from 'react'
import { api, type Facets, type Filters } from '../api'

interface Props {
  filters: Filters
  onChange: (filters: Filters) => void
  /** Photos matching the filters. */
  matching: number
  /** Changes when the library does, so the choices on offer stay current. */
  libraryPhotos: number
}

export function FilterBar({ filters, onChange, matching, libraryPhotos }: Props) {
  const [facets, setFacets] = useState<Facets | null>(null)

  useEffect(() => {
    let current = true
    api.facets().then(
      (f) => current && setFacets(f),
      () => {},
    )
    return () => {
      current = false
    }
  }, [libraryPhotos])

  const set = (patch: Filters) => {
    const next = { ...filters, ...patch }
    for (const key of Object.keys(next) as (keyof Filters)[]) {
      if (!next[key]) delete next[key]
    }
    onChange(next)
  }

  const cameras = facets?.cameras ?? []
  const scenes = facets?.scenes ?? []
  const occasions = facets?.occasions ?? []
  const people = facets?.people ?? []
  const active = Object.keys(filters).length > 0
  return (
    <div className="filters" role="search" aria-label="Filters">
      {(facets?.search_available || filters.q) && (
        // Keyed by the active query, so clearing the filters clears the box too.
        <SearchBox key={filters.q ?? ''} query={filters.q ?? ''} onSearch={(q) => set({ q })} />
      )}
      {(scenes.length > 0 || filters.scene) && (
        <label>
          <span>Scene</span>
          <select value={filters.scene ?? ''} onChange={(e) => set({ scene: e.target.value })}>
            <option value="">Any scene</option>
            {filters.scene && !scenes.some((s) => s.name === filters.scene) && (
              <option value={filters.scene}>{title(filters.scene)}</option>
            )}
            {scenes.map((s) => (
              <option key={s.name} value={s.name}>
                {title(s.name)} ({s.count.toLocaleString()})
              </option>
            ))}
          </select>
        </label>
      )}
      {(occasions.length > 0 || filters.occasion) && (
        <label>
          <span>Occasion</span>
          <select value={filters.occasion ?? ''} onChange={(e) => set({ occasion: e.target.value })}>
            <option value="">Any occasion</option>
            {filters.occasion && !occasions.some((o) => o.name === filters.occasion) && (
              <option value={filters.occasion}>{title(filters.occasion)}</option>
            )}
            {occasions.map((o) => (
              <option key={o.name} value={o.name}>
                {title(o.name)} ({o.count.toLocaleString()})
              </option>
            ))}
          </select>
        </label>
      )}
      {(people.length > 0 || filters.personId) && (
        <label>
          <span>Person</span>
          <select
            value={filters.personId ?? ''}
            onChange={(e) => set({ personId: Number(e.target.value) || undefined })}
          >
            <option value="">Everyone</option>
            {people.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} ({p.count.toLocaleString()})
              </option>
            ))}
          </select>
        </label>
      )}
      <label>
        <span>Camera</span>
        <select value={filters.camera ?? ''} onChange={(e) => set({ camera: e.target.value })}>
          <option value="">All cameras</option>
          {filters.camera && !cameras.some((c) => c.camera === filters.camera) && (
            <option value={filters.camera}>{filters.camera}</option>
          )}
          {cameras.map((c) => (
            <option key={c.camera} value={c.camera}>
              {c.camera} ({c.count.toLocaleString()})
            </option>
          ))}
        </select>
      </label>
      <label>
        <span>From</span>
        <input
          type="date"
          value={filters.dateFrom ?? ''}
          min={facets?.first_date ?? undefined}
          max={filters.dateTo ?? facets?.last_date ?? undefined}
          onChange={(e) => set({ dateFrom: e.target.value })}
        />
      </label>
      <label>
        <span>To</span>
        <input
          type="date"
          value={filters.dateTo ?? ''}
          min={filters.dateFrom ?? facets?.first_date ?? undefined}
          max={facets?.last_date ?? undefined}
          onChange={(e) => set({ dateTo: e.target.value })}
        />
      </label>
      {active && (
        <>
          <span className="muted filters-count">{matching.toLocaleString()} matching</span>
          <button onClick={() => onChange({})}>Clear filters</button>
        </>
      )}
    </div>
  )
}

function title(name: string): string {
  return name.charAt(0).toUpperCase() + name.slice(1)
}

function SearchBox({ query, onSearch }: { query: string; onSearch: (q: string) => void }) {
  const [text, setText] = useState(query)
  const submit = (e: FormEvent) => {
    e.preventDefault()
    onSearch(text.trim())
  }
  return (
    <form className="search" onSubmit={submit}>
      <input
        type="search"
        value={text}
        onChange={(e) => {
          setText(e.target.value)
          // Emptying the box removes the search at once.
          if (!e.target.value && query) onSearch('')
        }}
        placeholder="Search: friends in a pool, red shirt, a name…"
        aria-label="Search photos"
      />
      <button disabled={text.trim() === query}>Search</button>
    </form>
  )
}
