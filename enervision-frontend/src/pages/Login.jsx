import { useNavigate } from "react-router-dom";
import { useState } from "react";

/**
 * Login page layout:
 * - Left side: Login form (choose Admin or Resident)
 * - Right side: App info (what EnerVision does)
 */
export default function Login() {
  const nav = useNavigate();
  const [role, setRole] = useState("resident"); // default choice

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-emerald-50 via-white to-emerald-50">
      {/* Center the big card */}
      <div className="max-w-6xl mx-auto px-4 py-8 lg:py-12">
        <div className="bg-white/90 backdrop-blur rounded-2xl shadow-xl overflow-hidden grid grid-cols-1 lg:grid-cols-2">
          {/* LEFT: LOGIN PANEL */}
          <div className="p-6 sm:p-8 lg:p-10">
            {/* Brand */}
            <div className="flex items-center gap-3 mb-6">
              <span className="text-3xl">⚡</span>
              <h1 className="text-2xl sm:text-3xl font-bold text-emerald-700">EnerVision AI</h1>
            </div>

            {/* Welcome */}
            <p className="text-sm text-gray-600 mb-6">
              Welcome! Pick your role to enter the demo. No password needed.
            </p>

            {/* Role selector */}
            <div className="grid grid-cols-2 gap-3 mb-6">
              <button
                onClick={() => setRole("admin")}
                className={
                  "py-2 rounded-xl border text-sm " +
                  (role === "admin"
                    ? "border-emerald-600 bg-emerald-50 text-emerald-700"
                    : "hover:bg-gray-50")
                }
              >
                Apartment Admin
              </button>
              <button
                onClick={() => setRole("resident")}
                className={
                  "py-2 rounded-xl border text-sm " +
                  (role === "resident"
                    ? "border-emerald-600 bg-emerald-50 text-emerald-700"
                    : "hover:bg-gray-50")
                }
              >
                Resident
              </button>
            </div>

            {/* Optional fields (just for feel) */}
            <div className="space-y-3 mb-6">
              <input
                type="text"
                placeholder={role === "admin" ? "Building / Society Name" : "Flat / Unit (e.g., A-204)"}
                className="w-full rounded-xl border px-3 py-2 outline-none focus:ring-2 focus:ring-emerald-200"
              />
              <input
                type="text"
                placeholder="City (optional)"
                className="w-full rounded-xl border px-3 py-2 outline-none focus:ring-2 focus:ring-emerald-200"
              />
            </div>

            {/* Submit */}
            <button
              onClick={() => nav(role === "admin" ? "/admin" : "/resident")}
              className="w-full py-3 rounded-xl bg-emerald-600 text-white font-medium hover:bg-emerald-700 transition"
            >
              Enter Demo as {role === "admin" ? "Admin" : "Resident"}
            </button>

            {/* tiny note */}
            <p className="text-[11px] text-gray-500 mt-4">
              Prototype only — logins are mocked. You can change role anytime.
            </p>
          </div>

          {/* RIGHT: APP INFO PANEL */}
          <div className="relative bg-emerald-600/5">
            {/* Soft gradient ornament */}
            <div className="absolute -top-24 -right-24 h-64 w-64 rounded-full bg-emerald-200/40 blur-3xl" />
            <div className="absolute -bottom-24 -left-24 h-64 w-64 rounded-full bg-emerald-300/40 blur-3xl" />

            <div className="relative p-6 sm:p-8 lg:p-10 h-full flex flex-col">
              {/* Headline */}
              <h2 className="text-2xl font-semibold text-emerald-800">
                Smarter energy for apartments in India
              </h2>
              <p className="text-sm text-emerald-900/80 mt-2">
                Forecast usage, spot anomalies, and get AI tips to cut bills and carbon.
              </p>

              {/* Bullets */}
              <ul className="mt-6 space-y-3 text-sm text-emerald-900/90">
                <li className="flex gap-2">
                  <span>📈</span>
                  <span><b>Interactive Dashboard:</b> current vs building, peak hours, cost estimate.</span>
                </li>
                <li className="flex gap-2">
                  <span>🧠</span>
                  <span><b>AI Recommendations:</b> “Shift washing 9PM–6AM to save ~₹500/month.”</span>
                </li>
                <li className="flex gap-2">
                  <span>🌞</span>
                  <span><b>Solar & Battery ROI:</b> compare Solar vs Grid vs Storage.</span>
                </li>
                <li className="flex gap-2">
                  <span>🏘️</span>
                  <span><b>Community Insights:</b> see your rank vs building average.</span>
                </li>
              </ul>

              {/* Cute illustrative card */}
              <div className="mt-8 bg-white rounded-2xl shadow-lg p-4">
                <div className="text-sm text-gray-700 font-medium mb-2">Today’s peek</div>
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="bg-emerald-50 rounded-xl p-3">
                    <div className="text-xs text-gray-500">Peak</div>
                    <div className="text-base font-semibold text-emerald-700">18–22h</div>
                  </div>
                  <div className="bg-emerald-50 rounded-xl p-3">
                    <div className="text-xs text-gray-500">Your flat</div>
                    <div className="text-base font-semibold text-emerald-700">150 kWh</div>
                  </div>
                  <div className="bg-emerald-50 rounded-xl p-3">
                    <div className="text-xs text-gray-500">CO₂ saved</div>
                    <div className="text-base font-semibold text-emerald-700">120 kg</div>
                  </div>
                </div>
                <p className="text-[11px] text-gray-500 mt-3">
                  Demo values. Real data streams in from smart meters or CSV.
                </p>
              </div>

              {/* Footer note */}
              <div className="mt-auto pt-6 text-[11px] text-emerald-900/70">
                Built for India • Tariff-aware • Multilingual-ready • Gemini-powered chat
              </div>
            </div>
          </div>
        </div>

        {/* tiny footer */}
        <div className="text-center text-[11px] text-gray-500 mt-4">
          © {new Date().getFullYear()} EnerVision — Hackathon Prototype
        </div>
      </div>
    </div>
  );
}
