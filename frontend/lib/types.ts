// Mirrors backend/app/schemas.py. Update both together.

export type ReportStatus =
  | "submitted"
  | "no_damage"
  | "under_review"
  | "rejected"
  | "uav_dispatched"
  | "confirmed"
  | "not_confirmed";

export type PlanStatus = "proposed" | "approved" | "changes_needed" | "rejected";

export type MissionState = "pending" | "takeoff" | "surveying" | "landing" | "done" | "failed";

export interface Detection {
  id: number;
  source: "server" | "edge";
  deployment: string;
  cls: string;
  confidence: number;
  box_xyxy: [number, number, number, number];
  frame_id: string;
  model_version: string;
  lat: number | null;
  lon: number | null;
  size_m: number | null;
  annotated_url: string | null;
  flagged_wrong: boolean;
}

export interface Report {
  id: number;
  lat: number;
  lon: number;
  photo_url: string;
  status: ReportStatus;
  review_note: string | null;
  created_at: string;
  updated_at: string;
  detections: Detection[];
}

export interface Waypoint {
  lat: number;
  lon: number;
  alt: number;
  action: "takeoff" | "fly" | "survey" | "land";
}

export interface FlightPlan {
  id: number;
  report_id: number;
  drone_id: string;
  altitude_m: number;
  survey_radius_m: number;
  route: Waypoint[];
  status: PlanStatus;
  decision_reason: string | null;
  decided_at: string | null;
  created_at: string;
}

export interface Mission {
  id: number;
  flight_plan_id: number;
  state: MissionState;
  started_at: string | null;
  finished_at: string | null;
}

export interface Telemetry {
  id: number;
  mission_id: number;
  lat: number;
  lon: number;
  altitude_m: number;
  heading_deg: number;
  state: MissionState;
  ts: string;
}
