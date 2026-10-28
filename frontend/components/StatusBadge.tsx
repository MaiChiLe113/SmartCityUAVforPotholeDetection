import type { MissionState, PlanStatus, ReportStatus } from "@/lib/types";

const COLORS: Record<string, string> = {
  confirmed: "bg-green-100 text-green-800",
  approved: "bg-green-100 text-green-800",
  done: "bg-green-100 text-green-800",
  rejected: "bg-red-100 text-red-800",
  not_confirmed: "bg-red-100 text-red-800",
  no_damage: "bg-gray-100 text-gray-700",
  failed: "bg-red-100 text-red-800",
};

export default function StatusBadge({ status }: { status: ReportStatus | PlanStatus | MissionState }) {
  const color = COLORS[status] ?? "bg-blue-100 text-blue-800";
  return <span className={`rounded px-2 py-0.5 text-xs font-medium ${color}`}>{status.replace("_", " ")}</span>;
}
