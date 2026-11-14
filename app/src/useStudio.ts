import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type StudioJob, type StudioJobRequest, type StudioStatus } from './api'

/** Which models are installed. Read again whenever `revision` changes. */
export function useStudioStatus(revision = 0) {
  const [status, setStatus] = useState<StudioStatus | null>(null)
  useEffect(() => {
    let current = true
    api.studioStatus().then(
      (s) => current && setStatus(s),
      () => {},
    )
    return () => {
      current = false
    }
  }, [revision])
  return status
}

/**
 * One editing or generation job at a time: start it, follow its progress, and
 * cancel it. `onDone` is called once when a job finishes successfully.
 */
export function useJob(onDone?: (job: StudioJob) => void) {
  const [job, setJob] = useState<StudioJob | null>(null)
  const [error, setError] = useState<string | null>(null)
  const onDoneRef = useRef(onDone)
  useEffect(() => {
    onDoneRef.current = onDone
  }, [onDone])

  const active = job !== null && (job.state === 'queued' || job.state === 'running')
  const jobId = job?.id

  useEffect(() => {
    if (!active || jobId === undefined) return
    let current = true
    const id = setInterval(async () => {
      try {
        const next = await api.job(jobId)
        if (!current) return
        setJob(next)
        if (next.state === 'done') onDoneRef.current?.(next)
        if (next.state === 'error') setError(next.error ?? 'The job failed')
      } catch {
        // the service may be busy loading a model; keep asking
      }
    }, 500)
    return () => {
      current = false
      clearInterval(id)
    }
  }, [active, jobId])

  const start = useCallback(async (request: StudioJobRequest) => {
    setError(null)
    try {
      setJob(await api.submitJob(request))
    } catch (e) {
      setJob(null)
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  const cancel = useCallback(() => {
    if (jobId !== undefined) void api.cancelJob(jobId).then(setJob, () => {})
  }, [jobId])

  const reset = useCallback(() => {
    setJob(null)
    setError(null)
  }, [])

  return { job, active, error, start, cancel, reset }
}
