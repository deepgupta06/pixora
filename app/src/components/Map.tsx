import { useEffect, useRef, useState } from 'react'
import type { Filters, PhotoSummary } from '../api'
import { api } from '../api'
import maplibre from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'

interface Props {
  filters: Filters
  onShowPhoto: (photo: PhotoSummary) => void
}

const defaultLocation: [number, number] = [0, 20]
const defaultZoom = 1.5

export function Map({ filters: _filters, onShowPhoto: _onShowPhoto }: Props) {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<maplibre.Map | null>(null)
  const [noLocationPhotos] = useState<PhotoSummary[]>([])

  // Initialize map
  useEffect(() => {
    if (!mapContainer.current) return

    map.current = new maplibre.Map({
      container: mapContainer.current,
      style: 'https://demotiles.maplibre.org/style.json',
      center: defaultLocation,
      zoom: defaultZoom,
      attributionControl: false,
    })

    return () => {
      map.current?.remove()
    }
  }, [])

  return (
    <main className="map-container">
      <div ref={mapContainer} className="map-main" />

      {noLocationPhotos.length > 0 && (
        <div className="map-no-location">
          <h3>{noLocationPhotos.length} photos without location</h3>
          <div className="no-location-grid">
            {noLocationPhotos.slice(0, 12).map((photo) => (
              <div key={photo.id} className="no-location-thumb" onClick={() => _onShowPhoto(photo)}>
                <img src={api.thumbUrl(photo.hash, 256)} alt="No location" />
              </div>
            ))}
            {noLocationPhotos.length > 12 && (
              <div className="no-location-more">+{noLocationPhotos.length - 12}</div>
            )}
          </div>
        </div>
      )}
    </main>
  )
}
