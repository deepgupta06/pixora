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
 * Let the user put a file wherever they like, outside Pixora. Returns whether the file is
 * known to have been written: with the plain download that older browsers fall back to, the
 * browser does not say.
 */
export async function saveBlob(
  blob: Blob,
  { name, description, mime, ext }: SaveOptions,
): Promise<'written' | 'handed-over' | 'cancelled'> {
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
      throw e
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
