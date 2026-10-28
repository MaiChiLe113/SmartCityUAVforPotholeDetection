"use client";

// Leaflet map wrapper. Import it with next/dynamic and { ssr: false } (Leaflet needs `window`).
import "leaflet/dist/leaflet.css";

import L from "leaflet";
import { MapContainer, Marker, Polyline, Popup, TileLayer } from "react-leaflet";

import { MAP } from "@/lib/config";

// Default marker icons do not resolve under bundlers; point them at the CDN copies.
L.Icon.Default.mergeOptions({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

export interface MapMarker {
  id: string | number;
  position: [number, number];
  label?: string;
}

interface Props {
  center?: [number, number];
  zoom?: number;
  markers?: MapMarker[];
  route?: [number, number][];
  className?: string;
  children?: React.ReactNode; // e.g. <DraggablePin />
}

export default function MapView({
  center = MAP.defaultCenter,
  zoom = MAP.defaultZoom,
  markers = [],
  route,
  className = "h-96 w-full rounded",
  children,
}: Props) {
  return (
    <MapContainer center={center} zoom={zoom} className={className}>
      <TileLayer url={MAP.tileUrl} attribution={MAP.attribution} />
      {markers.map((m) => (
        <Marker key={m.id} position={m.position}>
          {m.label && <Popup>{m.label}</Popup>}
        </Marker>
      ))}
      {route && <Polyline positions={route} />}
      {children}
    </MapContainer>
  );
}
