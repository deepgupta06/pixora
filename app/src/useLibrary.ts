import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type Filters, type PhotoSummary, type ScanStatus, type Timeline } from './api'

const EMPTY: Timeline = { total: 0, buckets: [] }

/**
 * Library state for the grid. Only the months that have been scrolled into
 * view are held, and they are refreshed as the background scan adds photos.
 */
export function useLibrary(filters: Filters) {
  const [status, setStatus] = useState<ScanStatus | null>(null)
  const [timeline, setTimeline] = useState<Timeline>(EMPTY)
  const [months, setMonths] = useState<Record<string, PhotoSummary[]>>({})
  const [offline, setOffline] = useState(false)

  const monthsRef = useRef(months)
  const inFlight = useRef(new Set<string>())
  const lastPhotoCount = useRef(-1)
  const lastState = useRef<string | null>(null)
  // Bumped when the filters change, so answers to older requests are dropped.
  const generation = useRef(0)
  const filtersRef = useRef(filters)

  useEffect(() => {
    monthsRef.current = months
  }, [months])

  const loadMonth = useCallback(async (month: string, force = false) => {
    if (inFlight.current.has(month)) return
    if (!force && monthsRef.current[month]) return
    inFlight.current.add(month)
    const gen = generation.current
    try {
      const photos = await api.photos(month, filtersRef.current)
      if (gen === generation.current) setMonths((m) => ({ ...m, [month]: photos }))
    } catch {
      // the next status poll reports the service as offline
    } finally {
      if (gen === generation.current) inFlight.current.delete(month)
    }
  }, [])

  const refresh = useCallback(
    async (reloadAll: boolean) => {
      const gen = generation.current
      const next = await api.timeline(filtersRef.current)
      if (gen !== generation.current) return
      const counts = new Map(next.buckets.map((b) => [b.month, b.count]))
      setTimeline(next)
      setMonths((m) => {
        const kept = Object.entries(m).filter(([month]) => counts.has(month))
        return kept.length === Object.keys(m).length ? m : Object.fromEntries(kept)
      })
      for (const [month, photos] of Object.entries(monthsRef.current)) {
        const count = counts.get(month)
        if (count !== undefined && (reloadAll || count !== photos.length)) void loadMonth(month, true)
      }
    },
    [loadMonth],
  )

  const poll = useCallback(
    async (forceReload = false) => {
      try {
        const s = await api.status()
        setStatus(s)
        setOffline(false)
        const finished = lastState.current !== null && lastState.current !== 'idle' && s.state === 'idle'
        if (forceReload || finished || s.library.photos !== lastPhotoCount.current) {
          lastPhotoCount.current = s.library.photos
          await refresh(forceReload || finished)
        }
        lastState.current = s.state
      } catch {
        setOffline(true)
      }
    },
    [refresh],
  )

  // New filters: forget what was loaded and start from a fresh timeline.
  useEffect(() => {
    if (filtersRef.current === filters) return
    filtersRef.current = filters
    generation.current += 1
    inFlight.current.clear()
    monthsRef.current = {}
    const id = setTimeout(() => {
      setMonths({})
      void poll(true)
    }, 0)
    return () => clearTimeout(id)
  }, [filters, poll])

  const busy = !status || (status.state !== 'idle' && status.state !== 'paused')
  useEffect(() => {
    const first = setTimeout(() => void poll(), 0)
    const id = setInterval(() => void poll(), busy ? 1500 : 4000)
    return () => {
      clearTimeout(first)
      clearInterval(id)
    }
  }, [poll, busy])

  return { status, timeline, months, offline, loadMonth, poll }
}
