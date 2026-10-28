"use client";

// SCRUM-45 (Chi, backlog): report status that updates without reloading.
import { useParams } from "next/navigation";

import Todo from "@/components/Todo";

export default function ReportStatusPage() {
  const { id } = useParams<{ id: string }>();
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Report #{id}</h1>
      <Todo task="SCRUM-45, Chi">
        Poll getReport(id) every POLL_INTERVAL_MS.reportStatus; show submitted → under review → UAV dispatched →
        confirmed / not confirmed.
      </Todo>
    </div>
  );
}
