import { useState, useEffect } from 'react'
import RadarMap from './components/RadarMap'
import Header from './components/Header'
import Legend from './components/Legend'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [metadata, setMetadata] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)

  const fetchMetadata = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await fetch(`${API_URL}/radar/latest/metadata`)
      if (!response.ok) {
        throw new Error(`Failed to fetch metadata: ${response.statusText}`)
      }
      const data = await response.json()
      setMetadata(data)
      setLastUpdate(new Date())
    } catch (err) {
      setError(err.message)
      console.error('Error fetching metadata:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchMetadata()
    
    // Auto-refresh every 3 minutes
    const interval = setInterval(fetchMetadata, 3 * 60 * 1000)
    
    return () => clearInterval(interval)
  }, [])

  const handleRefresh = () => {
    fetchMetadata()
  }

  return (
    <div className="app">
      <Header 
        metadata={metadata}
        lastUpdate={lastUpdate}
        loading={loading}
        onRefresh={handleRefresh}
      />
      <div className="map-container">
        {error && (
          <div className="error-banner">
            <p>Error: {error}</p>
            <button onClick={handleRefresh}>Retry</button>
          </div>
        )}
        {loading && !metadata && (
          <div className="loading-overlay">
            <p>Loading radar data...</p>
          </div>
        )}
        {metadata && (
          <>
            <RadarMap metadata={metadata} apiUrl={API_URL} />
            <Legend />
          </>
        )}
      </div>
    </div>
  )
}

export default App

