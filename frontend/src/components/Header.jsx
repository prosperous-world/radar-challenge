function Header({ metadata, lastUpdate, loading, onRefresh }) {
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A'
    try {
      const date = new Date(timestamp)
      return date.toLocaleString('en-US', {
        timeZone: 'UTC',
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        timeZoneName: 'short'
      })
    } catch {
      return timestamp
    }
  }

  const formatLastUpdate = () => {
    if (!lastUpdate) return 'Never'
    const secondsAgo = Math.floor((new Date() - lastUpdate) / 1000)
    if (secondsAgo < 60) return `${secondsAgo}s ago`
    const minutesAgo = Math.floor(secondsAgo / 60)
    if (minutesAgo < 60) return `${minutesAgo}m ago`
    return formatTimestamp(lastUpdate)
  }

  return (
    <header className="header">
      <div className="header-content">
        <div className="header-title">
          <h1>MRMS Radar</h1>
          <p className="subtitle">Reflectivity at Lowest Altitude (RALA)</p>
        </div>
      
        <button 
          className="refresh-button" 
          onClick={onRefresh}
          disabled={loading}
        >
          {loading ? 'Loading...' : '🔄 Refresh'}
        </button>
      </div>
    </header>
  )
}

export default Header

