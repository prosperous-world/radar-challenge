import { useEffect, useRef } from 'react'
import { MapContainer, TileLayer, ImageOverlay } from 'react-leaflet'
import L from 'leaflet'

// Fix for default marker icons in React Leaflet
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

function RadarMap({ metadata, apiUrl }) {
  const imageUrl = `${apiUrl}/radar/latest/image?ts=${metadata.timestamp}`

  // Define bounds from metadata
  const bounds = [
    [metadata.south, metadata.west],  // Southwest corner
    [metadata.north, metadata.east],  // Northeast corner
  ]

  // Center point (roughly center of CONUS)
  const center = [
    (metadata.north + metadata.south) / 2,
    (metadata.east + metadata.west) / 2,
  ]

  return (
    <MapContainer
      center={center}
      zoom={4}
      style={{ height: '100%', width: '100%' }}
      scrollWheelZoom={true}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <ImageOverlay
        url={imageUrl}
        bounds={bounds}
        opacity={0.7}
        zIndex={100}
      />
    </MapContainer>
  )
}

export default RadarMap

