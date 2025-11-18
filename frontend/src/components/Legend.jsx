function Legend() {
  const dbzRanges = [
    { min: -10, max: 5, label: 'Light', color: 'rgba(0, 200, 0, 0.7)' },
    { min: 5, max: 20, label: 'Light-Moderate', color: 'rgba(255, 200, 0, 0.7)' },
    { min: 20, max: 35, label: 'Moderate', color: 'rgba(255, 150, 0, 0.8)' },
    { min: 35, max: 50, label: 'Heavy', color: 'rgba(255, 50, 0, 0.9)' },
    { min: 50, max: 70, label: 'Intense', color: 'rgba(200, 0, 100, 0.9)' },
  ]

  return (
    <div className="legend">
      <h3>Reflectivity (dBZ)</h3>
      <div className="legend-items">
        {dbzRanges.map((range, idx) => (
          <div key={idx} className="legend-item">
            <div 
              className="legend-color" 
              style={{ backgroundColor: range.color }}
            />
            <span className="legend-label">
              {range.min} to {range.max} dBZ
            </span>
            <span className="legend-desc">{range.label}</span>
          </div>
        ))}
      </div>
      <p className="legend-note">
        Data: NOAA MRMS • Updates every ~2 minutes
      </p>
    </div>
  )
}

export default Legend

