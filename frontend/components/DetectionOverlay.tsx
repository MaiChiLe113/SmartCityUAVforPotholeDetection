"use client";

// SCRUM-29 (Hai/Chi): photo with each detection box, class label and confidence drawn on top.
import type { Detection } from "@/lib/types";

import Todo from "./Todo";

interface Props {
  photoUrl: string;
  detections: Detection[];
}

export default function DetectionOverlay({ photoUrl, detections }: Props) {
  return (
    <Todo task="SCRUM-29">
      Draw {detections.length} box(es) over {photoUrl} (box_xyxy is in original image pixels).
    </Todo>
  );
}
