import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./index.css";

// PAGES
import Login from "./pages/Login.jsx";
// (we’ll add these in the next step)
import AdminDashboard from "./pages/AdminDashboard.jsx";
import ResidentDashboard from "./pages/ResidentDashboard.jsx";

const router = createBrowserRouter([
  { path: "/", element: <Login /> },
  { path: "/admin", element: <AdminDashboard /> },
  { path: "/resident", element: <ResidentDashboard /> },
]);

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
