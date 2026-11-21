import { createContext, useContext } from 'react'

export interface Locked {
  /** Ask for the key to a locked file, then show what is in it in a window of its own. */
  openFile: (file: File) => void
}

export const LockedContext = createContext<Locked | null>(null)

export function useLocked(): Locked {
  const locked = useContext(LockedContext)
  if (!locked) throw new Error('useLocked needs a LockedProvider')
  return locked
}
