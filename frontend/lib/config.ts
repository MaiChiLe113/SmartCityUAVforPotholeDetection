// Every changeable value in the frontend lives here (the backend equivalent is backend/config.py).

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const POLL_INTERVAL_MS = {
  reportStatus: 5000, // resident status page (SCRUM-45)
  telemetry: 1000, // dashboard live UAV position (SCRUM-39)
  reportQueue: 10000, // monitor report list
};

export const MAP = {
  defaultCenter: [10.7769, 106.7009] as [number, number], // used when geolocation is denied
  defaultZoom: 16,
  tileUrl: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  attribution: "&copy; OpenStreetMap contributors",
};

export const UPLOAD = {
  maxMb: 10, // keep in sync with backend config.API.max_upload_mb
  allowedTypes: ["image/jpeg", "image/png"],
};

export const CLASS_LABELS: Record<string, string> = {
  D00: "Longitudinal crack",
  D10: "Transverse crack",
  D20: "Alligator crack",
  D40: "Pothole",
};
