import { useEffect, useState, useRef } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import { useEffect as __useEffect } from 'react'

function App() {
  const [accuracy, setAccuracy] = useState('—')
  const [status, setStatus] = useState('')
  const [result, setResult] = useState(null)
  const fileRef = useRef()

  useEffect(() => {
    async function loadAccuracy() {
      try {
        const res = await fetch('http://127.0.0.1:5000/api/accuracy')
        const data = await res.json()
        setAccuracy(data.accuracy ?? 'N/A')
      } catch (e) {
        setAccuracy('N/A')
      }
    }
    loadAccuracy()
  }, [])

  async function uploadFile(e) {
    e.preventDefault()
    const file = fileRef.current.files[0]
    if (!file) return setStatus('Select a file first')

    // Client-side validation: ensure selected file is a video
    const isVideo = file.type && file.type.startsWith('video/')
    const allowedExt = ['.mp4', '.mov', '.avi', '.mkv']
    const lower = (file.name || '').toLowerCase()
    const extOk = allowedExt.some(e => lower.endsWith(e))
    if (!isVideo && !extOk) {
      return setStatus('Selected file is not a supported video type')
    }

    const fd = new FormData()
    fd.append('file', file)
    setStatus('Uploading...')
    try {
      const res = await fetch('http://127.0.0.1:5000/api/upload', {
        method: 'POST',
        body: fd
      })
      const data = await res.json()
      if (!res.ok) {
        setStatus('Upload failed')
        setResult(data)
      } else {
        setStatus('Upload successful')
        setResult(data)
      }
    } catch (err) {
      setStatus('Error: ' + err.message)
      setResult(null)
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <div>
          <a href="https://vite.dev" target="_blank" rel="noreferrer">
            <img src={viteLogo} className="logo" alt="Vite logo" />
          </a>
          <a href="https://react.dev" target="_blank" rel="noreferrer">
            <img src={reactLogo} className="logo react" alt="React logo" />
          </a>
        </div>
        <h1>Deepfake Video Detection — Frontend</h1>
      </header>

      <section className="panel">
        <h2>Model Accuracy</h2>
        <p>Current accuracy: <strong>{accuracy}</strong></p>
      </section>

      <section className="panel">
        <h2>Upload a video</h2>
        <form onSubmit={uploadFile}>
          <input ref={fileRef} type="file" accept="video/*" />
          <button type="submit">Upload</button>
        </form>
        <p>{status}</p>
        {result && (
          <pre style={{whiteSpace: 'pre-wrap', maxWidth: 800}}>
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
      </section>

      <section className="panel">
        <h2>Recent Results</h2>
        <ResultsList />
      </section>

      <footer style={{marginTop: 20}}>
        <small>Frontend served by Vite on :5173 — Backend expected on port 5000</small>
      </footer>
    </div>
  )
}

export default App


function ResultsList() {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true
    async function load() {
      try {
        const res = await fetch('http://127.0.0.1:5000/api/results')
        const data = await res.json()
        if (!mounted) return
        setItems(data.results || [])
      } catch (e) {
        console.error(e)
      } finally {
        if (mounted) setLoading(false)
      }
    }
    load()
    return () => { mounted = false }
  }, [])

  if (loading) return <p>Loading results…</p>
  if (!items || items.length === 0) return <p>No results yet.</p>

  return (
    <div>
      {items.map((it, idx) => (
        <div key={idx} style={{marginBottom:16}}>
          <div><strong>{it.name}</strong> — <em>{it.prediction}</em></div>
          {it.processed_video_url ? (
            <video controls width="480" style={{display:'block',marginTop:8}}>
              <source src={`http://127.0.0.1:5000${it.processed_video_url}`} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          ) : (
            <div style={{color:'#666', marginTop:8}}>No processed video available</div>
          )}
        </div>
      ))}
    </div>
  )
}
