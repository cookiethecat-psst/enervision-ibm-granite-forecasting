// src/pages/AdminDashboard.jsx
import React, { useEffect, useState } from "react";
import DashboardLayout from "../layouts/DashboardLayout.jsx";
import axios from "axios";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
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

export default function AdminDashboard() {
  const [forecast, setForecast] = useState([]);
  const [anomalies, setAnomalies] = useState([]);
  const [advisor, setAdvisor] = useState([]);
  const [selectedAnomaly, setSelectedAnomaly] = useState(null);
  const [popup, setPopup] = useState(null);

  const BASE_URL = "http://127.0.0.1:8000";

  useEffect(() => {
    axios.get(`${BASE_URL}/forecast`).then((res) => setForecast(res.data.forecast || []));
    axios.get(`${BASE_URL}/anomalies`).then((res) => setAnomalies(res.data.anomalies || []));
    axios.get(`${BASE_URL}/advisor`).then((res) => setAdvisor(res.data.tips || []));
  }, []);

  return (
    <DashboardLayout title="EnerVision — Admin" chatRole="admin">
      {/* KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <KPI label="Building usage (last 24h)" value={`${forecast.length} pts`} sub="Live data" />
        <KPI label="Anomalies detected" value={anomalies.length} sub="AI detection" />
        <KPI label="Tariff Window" value="18:00–22:00" sub="Peak pricing" />
      </div>

      {/* Forecast Chart */}
      <div className="relative bg-white rounded-2xl shadow-lg p-5 mt-4">
        <div className="text-xl font-semibold text-emerald-700 mb-1">Building Forecast</div>
        <div className="text-sm text-gray-500 mb-3">
          Last 24h kW readings. Red dots = anomalies (click to highlight). Red area = peak hours.
        </div>
        <div className="h-72">
          <ResponsiveContainer>
            <LineChart data={forecast}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip />
              <ReferenceArea x1="18:00" x2="22:00" fill="#FDE2E2" fillOpacity={0.6} />

              {/* Forecast line */}
              <Line type="monotone" dataKey="value" stroke="#007bff" dot={false} />

              {/* Anomaly dots */}
              <Line
                type="monotone"
                data={anomalies.map((a) => ({ time: a.time, anomaly: a.value }))}
                dataKey="anomaly"
                stroke="red"
                dot={(props) => {
                  const { cx, cy, payload } = props;
                  const isSelected =
                    selectedAnomaly && selectedAnomaly.time === payload.time;
                  return (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={isSelected ? 7 : 5}
                      fill={isSelected ? "darkred" : "red"}
                      stroke="white"
                      strokeWidth={1}
                      style={{ cursor: "pointer" }}
                      onClick={() => {
                        setSelectedAnomaly({ time: payload.time, value: payload.anomaly });
                        setPopup({
                          x: cx,
                          y: cy,
                          anomaly: anomalies.find((a) => a.time === payload.time),
                        });
                      }}
                    />
                  );
                }}
                strokeOpacity={0}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Popup */}
        {popup && (
          <div
            className="absolute bg-white shadow-lg rounded-lg p-2 text-xs"
            style={{ top: popup.y, left: popup.x + 60 }}
          >
            <div><b>Time:</b> {popup.anomaly.time}</div>
            <div><b>Value:</b> {popup.anomaly.value} kW</div>
            <div><b>Reason:</b> {popup.anomaly.reason}</div>
            <button
              className="text-red-500 text-[10px] mt-1"
              onClick={() => setPopup(null)}
            >
              Close
            </button>
          </div>
        )}
      </div>

      {/* Anomalies + Advisor */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
        <div className="bg-white rounded-2xl shadow-lg p-5">
          <div className="text-xl font-semibold text-emerald-700 mb-3">Anomalies</div>
          {anomalies.length === 0 ? (
            <p className="text-sm">✅ No anomalies detected in last 24h.</p>
          ) : (
            <ul className="space-y-2 text-sm">
              {anomalies.map((a, i) => {
                const isSelected = selectedAnomaly && selectedAnomaly.time === a.time;
                return (
                  <li
                    key={i}
                    className={`flex items-start gap-2 p-1 rounded cursor-pointer ${
                      isSelected ? "bg-red-100" : "hover:bg-gray-50"
                    }`}
                    onClick={() => setSelectedAnomaly(a)}
                  >
                    <span className="text-red-500">●</span>
                    <div>
                      <b>{a.time}</b> — {a.value} kW{" "}
                      <span className="text-gray-500">({a.reason})</span>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        <div className="bg-white rounded-2xl shadow-lg p-5">
          <div className="text-xl font-semibold text-emerald-700 mb-3">Efficiency Advisor</div>
          <ul className="space-y-2 text-sm">
            {advisor.map((tip, i) => (
              <li key={i}>
                <ReactMarkdown>{tip}</ReactMarkdown>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </DashboardLayout>
  );
}
