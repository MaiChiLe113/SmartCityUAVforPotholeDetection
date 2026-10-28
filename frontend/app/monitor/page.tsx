"use client";

// Report queue for the UAV monitor (SCRUM-23, SCRUM-29).
import Link from "next/link";
import { useEffect, useState } from "react";

import StatusBadge from "@/components/StatusBadge";
import { listReports } from "@/lib/api";
import type { Report } from "@/lib/types";

export default function MonitorPage() {
  const [reports, setReports] = useState<Report[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listReports()
      .then(setReports)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Reports</h1>
      {error && <p className="text-sm text-red-700">{error}</p>}
      <table className="w-full bg-white text-sm">
        <thead>
          <tr className="border-b text-left">
            <th className="p-2">#</th>
            <th className="p-2">Status</th>
            <th className="p-2">Top class</th>
            <th className="p-2">Confidence</th>
            <th className="p-2">Created</th>
          </tr>
        </thead>
        <tbody>
          {reports.map((r) => {
            const top = r.detections[0];
            return (
              <tr key={r.id} className="border-b">
                <td className="p-2">
                  <Link href={`/monitor/reports/${r.id}`} className="text-blue-700">
                    {r.id}
                  </Link>
                </td>
                <td className="p-2">
                  <StatusBadge status={r.status} />
                </td>
                <td className="p-2">{top?.cls ?? "-"}</td>
                <td className="p-2">{top ? top.confidence.toFixed(2) : "-"}</td>
                <td className="p-2">{new Date(r.created_at).toLocaleString()}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
