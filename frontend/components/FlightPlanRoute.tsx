"use client";

// SCRUM-31 (Hoan): plan route on the map with takeoff, target pin, survey area and landing.
import dynamic from "next/dynamic";

import type { FlightPlan } from "@/lib/types";

import Todo from "./Todo";

const MapView = dynamic(() => import("./MapView"), { ssr: false });

export default function FlightPlanRoute({ plan }: { plan: FlightPlan }) {
  const route = plan.route.map((w) => [w.lat, w.lon] as [number, number]);
  return (
    <div className="space-y-2">
      <MapView center={route[0]} route={route} />
      <Todo task="SCRUM-31, Hoan">Mark takeoff/landing, draw the survey circle (plan.survey_radius_m).</Todo>
    </div>
  );
}
