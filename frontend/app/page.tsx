import Link from "next/link";

export default function Home() {
  return (
    <div className="mt-16 flex flex-col items-center gap-6 text-center">
      <h1 className="text-2xl font-semibold">Smart City UAV for Pothole Detection</h1>
      <p className="max-w-md text-slate-600">
        Residents pin a pothole and send a photo. The UAV monitor reviews it, approves a flight plan and
        follows the drone as it confirms the damage.
      </p>
      <div className="flex gap-4">
        <Link href="/report" className="rounded bg-blue-600 px-4 py-2 text-white">
          Report a pothole
        </Link>
        <Link href="/monitor" className="rounded border px-4 py-2">
          UAV monitor
        </Link>
      </div>
    </div>
  );
}
