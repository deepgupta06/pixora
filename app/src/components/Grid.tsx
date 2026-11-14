import { useVirtualizer } from '@tanstack/react-virtual'
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { OriginBadge } from './StudioParts'
import { api, type PhotoSummary, type Timeline } from '../api'
import { buildRows, GAP, monthLabel } from '../layout'

interface Props {
  timeline: Timeline
  months: Record<string, PhotoSummary[]>
  loadMonth: (month: string) => void
  onOpen: (photo: PhotoSummary) => void
}

export function Grid({ timeline, months, loadMonth, onOpen }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const innerRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)

  useLayoutEffect(() => {
    const el = innerRef.current
    if (!el) return
    const observer = new ResizeObserver(() => setWidth(Math.floor(el.clientWidth)))
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const rows = useMemo(() => buildRows(timeline, months, width), [timeline, months, width])

  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: (i) => rows[i].height,
    getItemKey: (i) => rows[i].key,
    overscan: 6,
  })

  // Row heights are exact and change whenever the layout does.
  useLayoutEffect(() => {
    virtualizer.measure()
  }, [rows, virtualizer])

  const items = virtualizer.getVirtualItems()

  // Fetch a month only when its placeholder scrolls into range.
  useEffect(() => {
    for (const item of items) {
      const row = rows[item.index]
      if (row?.type === 'placeholder') loadMonth(row.month)
    }
  })

  return (
    <div className="grid-scroll" ref={scrollRef}>
      <div className="grid-inner" ref={innerRef} style={{ height: virtualizer.getTotalSize() }}>
        {items.map((item) => {
          const row = rows[item.index]
          if (!row) return null
          const style = { transform: `translateY(${item.start}px)`, height: row.height }
          if (row.type === 'header') {
            return (
              <div key={item.key} className="grid-row grid-header" style={style}>
                <h2>{monthLabel(row.month)}</h2>
                <span>{row.count.toLocaleString()}</span>
              </div>
            )
          }
          if (row.type === 'placeholder') {
            return <div key={item.key} className="grid-row grid-placeholder" style={style} />
          }
          return (
            <div key={item.key} className="grid-row grid-photos" style={{ ...style, gap: GAP }}>
              {row.items.map(({ photo, width: w }) => (
                <button
                  key={photo.id}
                  className="tile"
                  style={{ width: w, height: row.rowHeight }}
                  onClick={() => onOpen(photo)}
                  aria-label={`Photo taken ${new Date(photo.taken_at).toLocaleString()}`}
                >
                  <img src={api.thumbUrl(photo.hash, 256)} alt="" decoding="async" draggable={false} />
                  <OriginBadge origin={photo.origin} />
                </button>
              ))}
            </div>
          )
        })}
      </div>
    </div>
  )
}
