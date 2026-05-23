// hooks/useDatasets.js — GET /datasets/status + POST /datasets/update (SSE)

const API = '/api'

export function useDatasets() {
  /**
   * Fetch current dataset status for all configured languages.
   * Returns { languages: { en: { records, last_updated, png_cached } } }
   */
  async function getStatus() {
    const res = await fetch(`${API}/datasets/status`)
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  /**
   * Start a dataset update job.
   * Streams SSE log lines via onLog(msg: string) callback.
   * Resolves when the stream ends with { ok: bool }.
   *
   * @param {object}   opts
   * @param {string[]} opts.langs            – languages to update (default: all)
   * @param {boolean}  opts.force            – force re-fetch even if up to date
   * @param {boolean}  opts.download_images  – also download PNG files
   * @param {function} opts.onLog            – called for each log line received
   */
  function startUpdate({ langs, force = false, download_images = false, onLog }) {
    return new Promise(async (resolve, reject) => {
      let res
      try {
        res = await fetch(`${API}/datasets/update`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ langs, force, download_images }),
        })
      } catch (e) {
        reject(e)
        return
      }

      if (res.status === 409) {
        reject(new Error('An update job is already running'))
        return
      }
      if (!res.ok) {
        reject(new Error(await res.text()))
        return
      }

      // Parse SSE stream manually (fetch + ReadableStream)
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      const pump = async () => {
        while (true) {
          let done, value
          try {
            ;({ done, value } = await reader.read())
          } catch (e) {
            reject(e)
            return
          }
          if (done) {
            resolve({ ok: true })
            return
          }
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() // keep incomplete line
          for (const line of lines) {
            if (!line.startsWith('data:')) continue
            try {
              const event = JSON.parse(line.slice(5).trim())
              if (event.type === 'log' && onLog) {
                onLog(event.msg)
              } else if (event.type === 'done') {
                resolve({ ok: event.ok ?? true })
                reader.cancel()
                return
              }
            } catch (_) {
              // malformed line — skip
            }
          }
        }
      }

      pump()
    })
  }

  return { getStatus, startUpdate }
}
