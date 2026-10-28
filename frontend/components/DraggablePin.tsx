"use client";

// SCRUM-24 (Chi): a marker the resident drags to the pothole; reports the new position.
import type { LatLngTuple } from "leaflet";

interface Props {
  position: LatLngTuple;
  onChange: (position: LatLngTuple) => void;
}

export default function DraggablePin({ position, onChange }: Props) {
  // TODO(SCRUM-24, Chi): <Marker draggable position={position}
  //   eventHandlers={{ dragend: (e) => onChange(...) }} />
  void position;
  void onChange;
  return null;
}
