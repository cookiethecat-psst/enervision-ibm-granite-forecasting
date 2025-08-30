// src/App.js
import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";

// PAGES (from what we created)
import Login from "./pages/Login.jsx";
import AdminDashboard from "./pages/AdminDashboard.jsx";
// We'll add Resident next, but include the route now.
import ResidentDashboard from "./pages/ResidentDashboard.jsx";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/admin" element={<AdminDashboard />} />
        <Route path="/resident" element={<ResidentDashboard />} />
      </Routes>
    </BrowserRouter>
  );
}
