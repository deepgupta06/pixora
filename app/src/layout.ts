import type { PhotoSummary, Timeline } from './api'

export const TARGET_ROW_HEIGHT = 190
export const GAP = 4
export const HEADER_HEIGHT = 56

export interface RowItem {
  photo: PhotoSummary
  width: number
}

export type Row =
  | { type: 'header'; key: string; month: string; count: number; height: number }
  | { type: 'photos'; key: string; month: string; height: number; rowHeight: number; items: RowItem[] }
  | { type: 'placeholder'; key: string; month: string; height: number }

function aspect(p: PhotoSummary): number {
  // Clamp extremes so one panorama or sliver cannot wreck a row.
  return Math.min(3.2, Math.max(0.45, p.width / p.height))
}

/** Fill rows left to right, then scale each row to the full width. */
export function justify(photos: PhotoSummary[], width: number): { height: number; items: RowItem[] }[] {
  const rows: { height: number; items: RowItem[] }[] = []
  let current: PhotoSummary[] = []
  let aspectSum = 0
  const emit = (height: number) => {
    rows.push({ height, items: current.map((photo) => ({ photo, width: aspect(photo) * height })) })
    current = []
    aspectSum = 0
  }
  for (const photo of photos) {
    current.push(photo)
    aspectSum += aspect(photo)
    const height = (width - GAP * (current.length - 1)) / aspectSum
    if (height <= TARGET_ROW_HEIGHT) emit(height)
  }
  // The last row keeps the target height instead of stretching a few photos.
  if (current.length) emit(TARGET_ROW_HEIGHT)
  return rows
}

function estimateHeight(count: number, width: number): number {
  const perRow = Math.max(1, Math.round(width / (TARGET_ROW_HEIGHT * 1.35 + GAP)))
  return Math.ceil(count / perRow) * (TARGET_ROW_HEIGHT + GAP)
}

/** Flatten the timeline into virtual rows. Months not loaded yet become one placeholder. */
export function buildRows(
  timeline: Timeline,
  months: Record<string, PhotoSummary[]>,
  width: number,
): Row[] {
  if (width <= 0) return []
  const rows: Row[] = []
  for (const { month, count } of timeline.buckets) {
    const photos = months[month]
    rows.push({ type: 'header', key: `h-${month}`, month, count, height: HEADER_HEIGHT })
    if (!photos) {
      rows.push({ type: 'placeholder', key: `p-${month}`, month, height: estimateHeight(count, width) })
      continue
    }
    for (const row of justify(photos, width)) {
      rows.push({
        type: 'photos',
        key: `r-${month}-${row.items[0].photo.id}`,
        month,
        height: row.height + GAP,
        rowHeight: row.height,
        items: row.items,
      })
    }
  }
  return rows
}

export function monthLabel(month: string): string {
  const [y, m] = month.split('-').map(Number)
  return new Date(y, m - 1, 1).toLocaleDateString(undefined, { month: 'long', year: 'numeric' })
}
