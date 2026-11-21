export interface SaveOptions {
  /** What the save window suggests. */
  name: string
  description: string
  mime: string
  ext: string
}

type Writable = { write: (data: Blob) => Promise<void>; close: () => Promise<void> }
type SavePicker = (options: object) => Promise<{ createWritable: () => Promise<Writable> }>

/**
 * What came of saving: written where the user chose; handed to the browser's own downloads,
 * which does not say whether it worked; cancelled by the user; or not tried yet, because the
 * browser wants a fresh click first (see `saveBlob`).
 */
export type SaveResult = 'written' | 'handed-over' | 'cancelled' | 'needs-click'

/**
 * Let the user put a file wherever they like, outside Pixora.
 *
 * The browser opens its save window only for a click it saw in the last few seconds, and
 * making a file can take longer than that. When it refuses for that reason this gives
 * 'needs-click', and the caller asks for another click and tries again with `fromClick`. A
 * call made straight from a click that still gets refused falls back to a plain download.
 */
export async function saveBlob(
  blob: Blob,
  { name, description, mime, ext }: SaveOptions,
  fromClick = false,
): Promise<SaveResult> {
  const picker = (window as unknown as { showSaveFilePicker?: SavePicker }).showSaveFilePicker
  if (picker) {
    try {
      const handle = await picker({ suggestedName: name, types: [{ description, accept: { [mime]: [ext] } }] })
      const file = await handle.createWritable()
      await file.write(blob)
      await file.close()
      return 'written'
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return 'cancelled'
      if (!(e instanceof DOMException && e.name === 'SecurityError')) throw e
      if (!fromClick) return 'needs-click'
    }
  }
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = name
  link.click()
  setTimeout(() => URL.revokeObjectURL(link.href), 10_000)
  return 'handed-over'
}

export function base64ToBlob(data: string, type: string): Blob {
  return new Blob([Uint8Array.from(atob(data), (c) => c.charCodeAt(0))], { type })
}

export function fileToBase64(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result).split(',')[1] ?? '')
    reader.onerror = () => reject(reader.error ?? new Error('Could not read the file'))
    reader.readAsDataURL(file)
  })
}
