// src/index.js
import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./index.css";

// PAGES
import Login from "./pages/Login.jsx";
import AdminDashboard from "./pages/AdminDashboard.jsx";
import ResidentDashboard from "./pages/ResidentDashboard.jsx";

const router = createBrowserRouter([
  { path: "/", element: <Login /> },
  { path: "/admin", element: <AdminDashboard /> },
  { path: "/resident", element: <ResidentDashboard /> },
]);

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
