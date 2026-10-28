"use client";

// SCRUM-24 / SCRUM-30 (Chi): map at the resident's location, draggable pin, photo, submit.
import dynamic from "next/dynamic";
import Link from "next/link";
import { useState } from "react";

import PhotoUpload from "@/components/PhotoUpload";
import Todo from "@/components/Todo";
import { createReport } from "@/lib/api";
import { MAP } from "@/lib/config";
import type { Report } from "@/lib/types";

const MapView = dynamic(() => import("@/components/MapView"), { ssr: false });

export default function ReportPage() {
  const [position] = useState<[number, number]>(MAP.defaultCenter);
  const [photo, setPhoto] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [created, setCreated] = useState<Report | null>(null);

  // TODO(SCRUM-24, Chi): start at navigator.geolocation position; update via <DraggablePin>.

  async function submit() {
    if (!photo) return setError("Add a photo first.");
    setError(null);
    try {
      setCreated(await createReport(position[0], position[1], photo));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Report a pothole</h1>
      <MapView center={position} markers={[{ id: "pin", position }]} />
      <Todo task="SCRUM-24, Chi">Open at current location and make the pin draggable.</Todo>
      <PhotoUpload onSelect={setPhoto} />
      <button onClick={submit} className="rounded bg-blue-600 px-4 py-2 text-white">
        Send report
      </button>
      {error && <p className="text-sm text-red-700">{error}</p>}
      {created && (
        <p className="text-sm">
          Report sent. <Link href={`/report/${created.id}`}>Follow its status</Link>
        </p>
      )}
    </div>
  );
}
