"use client";

// SCRUM-30 (Chi): pick a JPEG/PNG, check type/size from lib/config UPLOAD, show a preview.
import Todo from "./Todo";

interface Props {
  onSelect: (file: File | null) => void;
}

export default function PhotoUpload({ onSelect }: Props) {
  void onSelect;
  return <Todo task="SCRUM-30, Chi">File input + preview; reject other types or files over UPLOAD.maxMb.</Todo>;
}
