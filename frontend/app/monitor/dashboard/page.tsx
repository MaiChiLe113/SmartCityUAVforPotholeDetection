"use client";

// SCRUM-39 (Chi): live UAV position, mission state and target pothole on one map.
import dynamic from "next/dynamic";

import Todo from "@/components/Todo";

const MapView = dynamic(() => import("@/components/MapView"), { ssr: false });

export default function DashboardPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">UAV dashboard</h1>
      <MapView className="h-[32rem] w-full rounded" />
      <Todo task="SCRUM-39, Chi">
        listMissions(true), then poll latestTelemetry(missionId) every POLL_INTERVAL_MS.telemetry; show the UAV marker,
        state (takeoff / surveying / landing) and the target pin; show confirmed / not confirmed when done.
      </Todo>
    </div>
  );
}
