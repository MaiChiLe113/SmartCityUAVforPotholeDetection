"use client";

// SCRUM-29 / SCRUM-37: detection result, confirm/reject the report, flag a wrong detection.
import { useParams } from "next/navigation";

import Todo from "@/components/Todo";

export default function ReviewReportPage() {
  const { id } = useParams<{ id: string }>();
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Review report #{id}</h1>
      <Todo task="SCRUM-29">
        getReport(id) → &lt;DetectionOverlay&gt; with class, confidence, frame id and model version per box.
      </Todo>
      <Todo task="SCRUM-29, Chi">
        Confirm → reviewReport(id, true) then proposeFlightPlan(id) and go to /monitor/flight-plans/[planId]. Reject →
        reviewReport(id, false, note).
      </Todo>
      <Todo task="SCRUM-37, Chi">&quot;Wrong detection&quot; button per box → flagDetection(detectionId, note).</Todo>
    </div>
  );
}
