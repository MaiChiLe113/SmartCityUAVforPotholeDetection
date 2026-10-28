// One typed wrapper per backend endpoint. Unbuilt endpoints answer 501 with
// "Not implemented yet: SCRUM-xx", which surfaces as the thrown error message.

import { API_URL } from "./config";
import type { Detection, FlightPlan, Mission, Report, Telemetry } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

const json = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const mediaUrl = (path: string) => (path.startsWith("http") ? path : `${API_URL}${path}`);

// Reports (SCRUM-23, 29, 30, 45)
export function createReport(lat: number, lon: number, photo: File): Promise<Report> {
  const form = new FormData();
  form.append("lat", String(lat));
  form.append("lon", String(lon));
  form.append("photo", photo);
  return request("/reports", { method: "POST", body: form });
}
export const listReports = (status?: string) =>
  request<Report[]>(`/reports${status ? `?status=${status}` : ""}`);
export const getReport = (id: number) => request<Report>(`/reports/${id}`);
export const reviewReport = (id: number, confirmed: boolean, note?: string) =>
  request<Report>(`/reports/${id}/review`, json("PATCH", { confirmed, note }));

// Detections (SCRUM-29, 37)
export const listDetections = (reportId: number) =>
  request<Detection[]>(`/reports/${reportId}/detections`);
export const flagDetection = (id: number, note?: string) =>
  request<Detection>(`/detections/${id}/flag`, json("POST", { note }));

// Flight plans (SCRUM-27, 31)
export const proposeFlightPlan = (reportId: number) =>
  request<FlightPlan>(`/reports/${reportId}/flight-plan`, json("POST"));
export const getFlightPlan = (id: number) => request<FlightPlan>(`/flight-plans/${id}`);
export const approveFlightPlan = (id: number) =>
  request<FlightPlan>(`/flight-plans/${id}/approve`, json("POST"));
export const rejectFlightPlan = (id: number, reason: string) =>
  request<FlightPlan>(`/flight-plans/${id}/reject`, json("POST", { reason }));
export const requestPlanChanges = (id: number, reason: string) =>
  request<FlightPlan>(`/flight-plans/${id}/request-changes`, json("POST", { reason }));

// Missions (SCRUM-33, 39)
export const listMissions = (active = false) => request<Mission[]>(`/missions?active=${active}`);
export const dispatchMission = (id: number) =>
  request<Mission>(`/missions/${id}/dispatch`, json("POST"));
export const latestTelemetry = (missionId: number) =>
  request<Telemetry>(`/missions/${missionId}/telemetry/latest`);
