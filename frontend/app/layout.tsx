import type { Metadata } from "next";
import Link from "next/link";

import "./globals.css";

export const metadata: Metadata = {
  title: "Pothole UAV",
  description: "Report potholes and follow UAV confirmation",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <nav className="flex gap-6 border-b bg-white px-4 py-3 text-sm font-medium">
          <Link href="/" className="font-semibold">
            Pothole UAV
          </Link>
          <Link href="/report">Report a pothole</Link>
          <Link href="/monitor">Monitor</Link>
          <Link href="/monitor/dashboard">Dashboard</Link>
        </nav>
        <main className="mx-auto max-w-5xl p-4">{children}</main>
      </body>
    </html>
  );
}
