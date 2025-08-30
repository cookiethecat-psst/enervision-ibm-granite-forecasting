// src/pages/ResidentDashboard.jsx
import React, { useEffect, useState } from "react";
import DashboardLayout from "../layouts/DashboardLayout.jsx";
import axios from "axios";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceArea,
} from "recharts";
import ReactMarkdown from "react-markdown";

function KPI({ label, value, sub }) {
  return (
    <div className="bg-white rounded-2xl shadow-lg p-5">
      <div className="text-sm text-gray-500">{label}</div>
      <div className="text-2xl font-semibold mt-1">{value}</div>
      {sub && <div className="text-xs text-gray-500 mt-1">{sub}</div>}
    </div>
  );
}

/** Demo leaderboard rows (can connect later) */
const ranks = [
  { name: "You", kwh: 150 },
  { name: "A-204", kwh: 160 },
  { name: "C-402", kwh: 170 },
  { name: "B-307", kwh: 180 },
  { name: "D-101", kwh: 210 },
];

export default function ResidentDashboard() {
  const [forecast, setForecast] = useState([]);
  const [advisor, setAdvisor] = useState([]);

  const BASE_URL = "http://127.0.0.1:8000"; // backend

  useEffect(() => {
    axios.get(`${BASE_URL}/forecast`).then((res) => setForecast(res.data.forecast || []));
    axios.get(`${BASE_URL}/advisor`).then((res) => setAdvisor(res.data.tips || []));
  }, []);

  return (
    <DashboardLayout title="EnerVision — Resident" chatRole="resident">
      {/* KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <KPI label="Your flat — 24h" value={`${forecast.length} pts`} sub="Live usage data" />
        <KPI label="Peak hours" value="18:00–22:00" sub="Shift laundry/dishwasher" />
        <KPI label="Est. monthly cost" value="₹3,450" sub="Assuming ₹7/kWh tariff" />
      </div>

      {/* Chart + Leaderboard */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
        {/* Chart */}
        <div className="lg:col-span-2 bg-white rounded-2xl shadow-lg p-5">
          <div className="text-xl font-semibold text-emerald-700 mb-1">Your Usage (last 24h)</div>
          <div className="text-sm text-gray-500 mb-3">Red = peak hours.</div>
          <div className="h-72">
            <ResponsiveContainer>
              <LineChart data={forecast}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <ReferenceArea x1="18:00" x2="22:00" fill="#FDE2E2" fillOpacity={0.6} />
                <Line type="monotone" dataKey="value" stroke="#28a745" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Leaderboard */}
        <div className="bg-white rounded-2xl shadow-lg p-5">
          <div className="text-xl font-semibold text-emerald-700 mb-3">Community Leaderboard</div>
          <div className="text-xs text-gray-500 mb-2">Lower kWh = more efficient</div>
          <div className="divide-y">
            {ranks.sort((a, b) => a.kwh - b.kwh).map((r, i) => (
              <div key={r.name} className="py-2 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-6 text-gray-500">{i + 1}</div>
                  <div className={r.name === "You" ? "font-semibold text-emerald-700" : ""}>
                    {r.name}
                  </div>
                </div>
                <div className="text-sm">{r.kwh} kWh</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Advisor Tips */}
      <div className="bg-white rounded-2xl shadow-lg p-5 mt-6">
        <div className="text-xl font-semibold text-emerald-700 mb-3">Personalized Tips</div>
        <ul className="space-y-2 text-sm">
          {advisor.map((tip, i) => (
            <li key={i}>
              <ReactMarkdown>{tip}</ReactMarkdown>
            </li>
          ))}
        </ul>
      </div>
    </DashboardLayout>
  );
}
