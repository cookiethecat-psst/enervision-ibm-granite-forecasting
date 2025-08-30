// src/layouts/DashboardLayout.jsx
import React from "react";
import ChatPanel from "../shared/ChatPanel.jsx";

/**
 * Two-column layout:
 * - Left: page content (KPIs, charts, tables)
 * - Right: sticky chat assistant
 *
 * Pass chatRole="admin" or "resident"
 */
export default function DashboardLayout({ children, title = "EnerVision AI", chatRole = "resident" }) {
  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-emerald-50 via-white to-emerald-50">
      {/* Header */}
      <header className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-2xl">⚡</span>
            <h1 className="text-xl font-semibold text-emerald-700">{title}</h1>
          </div>

          <nav className="hidden md:flex gap-6 text-sm">
            <span className="hover:text-emerald-700 cursor-pointer">Dashboard</span>
            <span className="hover:text-emerald-700 cursor-pointer">Advisor</span>
            <span className="hover:text-emerald-700 cursor-pointer">Reports</span>
            <span className="hover:text-emerald-700 cursor-pointer">Settings</span>
          </nav>

          <div className="flex gap-2">
            <button className="px-3 py-1.5 rounded-full bg-emerald-600 text-white">Try Demo</button>
            <button className="px-3 py-1.5 rounded-full border">Docs</button>
          </div>
        </div>
      </header>

      {/* Content + Chat */}
      <div className="max-w-6xl mx-auto px-4 py-6 grid gap-6 lg:grid-cols-[1fr_360px]">
        <main className="space-y-6">{children}</main>
        <aside className="lg:sticky lg:top-20 h-[72vh]">
          <ChatPanel role={chatRole} />
        </aside>
      </div>

      <footer className="text-center text-[11px] text-gray-500 pb-6">
        © {new Date().getFullYear()} EnerVision — Dashboard
      </footer>
    </div>
  );
}
