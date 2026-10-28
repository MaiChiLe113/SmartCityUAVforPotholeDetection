"use client";

// SCRUM-31 (Hoan): review and approve or reject a proposed flight plan. Reject needs a reason.
import { useParams } from "next/navigation";

import Todo from "@/components/Todo";

export default function FlightPlanPage() {
  const { id } = useParams<{ id: string }>();
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Flight plan #{id}</h1>
      <Todo task="SCRUM-31, Hoan">
        getFlightPlan(id) → &lt;FlightPlanRoute plan=… /&gt;, altitude, drone, status.
      </Todo>
      <Todo task="SCRUM-31, Hoan">
        Buttons: Approve → approveFlightPlan(id); Request changes / Reject → reason textbox (required) →
        requestPlanChanges / rejectFlightPlan.
      </Todo>
    </div>
  );
}
