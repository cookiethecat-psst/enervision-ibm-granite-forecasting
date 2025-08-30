// src/api/generate.js (CRA) or any component file
const API_BASE =
  process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

export async function generateText(prompt) {
  try {
    const res = await fetch(`${API_BASE}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        temperature: 0.2,
        max_output_tokens: 256,
      }),
    });

    if (!res.ok) {
      // Read error body for details from FastAPI
      const err = await res.json().catch(() => ({}));
      const msg = err?.detail || `HTTP ${res.status}`;
      throw new Error(`Backend error: ${msg}`);
    }

    const data = await res.json();
    return data.text; // { text: "..." }
  } catch (err) {
    // Distinguish network vs backend errors
    if (err instanceof TypeError) {
      // Often indicates CORS or server unreachable
      throw new Error("Network error: Unable to reach backend (check URL/CORS).");
    }
    throw err;
  }
}
