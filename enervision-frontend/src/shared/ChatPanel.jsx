// src/shared/ChatPanel.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// CRA uses REACT_APP_* env; fallback to localhost:8000
const API_BASE = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

export default function ChatPanel({
  role = "resident",
  buildingId = "BLDG-01",
  flatId = "A-204",
  context = null,
}) {
  // rotating “Try:” suggestions based on role
  const suggestions = useMemo(
    () =>
      role === "admin"
        ? [
            "What’s today’s peak and 3 actions to shave 15%?",
            "Rough ROI for 35 kW solar + 40 kWh battery?",
            "Which anomalies should we fix first today?",
            "How to schedule pumps/EVs to avoid 18–22h?",
          ]
        : [
            "How can I cut my monthly bill by 20% based on today’s forecast?",
            "What should I move out of 18–22h today?",
            "Is my AC usage too high vs building?",
            "Give me 5 tailored tips for tonight.",
          ],
    [role]
  );

  const [hintIndex, setHintIndex] = useState(0);
  useEffect(() => {
    const id = setInterval(
      () => setHintIndex((i) => (i + 1) % suggestions.length),
      7000
    );
    return () => clearInterval(id);
  }, [suggestions]);

  const [messages, setMessages] = useState([
    {
      role: "ai",
      text: "Hi! I’m EnerVision. I’ll use today’s data to answer your questions.",
    },
    { role: "ai", text: `Try: “${suggestions[0]}”` },
  ]);

  // update the rotating hint bubble when index changes
  useEffect(() => {
    setMessages((m) => {
      const cloned = [...m];
      if (
        cloned.length >= 2 &&
        cloned[1]?.role === "ai" &&
        cloned[1].text.startsWith("Try:")
      ) {
        cloned[1] = { role: "ai", text: `Try: “${suggestions[hintIndex]}”` };
      }
      return cloned;
    });
  }, [hintIndex, suggestions]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send() {
    const t = input.trim();
    if (!t || loading) return;
    setMessages((m) => [...m, { role: "user", text: t }]);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: t,
          role,
          building_id: buildingId,
          flat_id: flatId,
          context,
        }),
      });

      if (!res.ok) {
        const text = await res.text();
        setMessages((m) => [
          ...m,
          { role: "ai", text: `⚠️ Server error ${res.status}: ${text}` },
        ]);
      } else {
        const data = await res.json();
        const reply = data?.reply?.trim();
        setMessages((m) => [
          ...m,
          { role: "ai", text: reply || "I received an empty reply." },
        ]);
      }
    } catch (e) {
      setMessages((m) => [
        ...m,
        {
          role: "ai",
          text: `⚠️ Network error: ${String(
            e
          )}. Is the backend running on ${API_BASE}?`,
        },
      ]);
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="h-full flex flex-col bg-white rounded-2xl shadow-lg p-4">
      {/* Header */}
      <div className="font-semibold mb-1">Ask EnerVision</div>
      <div className="text-xs text-gray-500 mb-3">
        Role: {role} • Answers grounded in today’s data
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-auto space-y-2 pr-1">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`max-w-[84%] break-words ${
              m.role === "ai"
                ? "bg-gray-100 text-gray-800"
                : "bg-emerald-600 text-white ml-auto"
            } rounded-2xl px-3 py-2 text-sm`}
          >
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown>
          </div>
        ))}
        {loading && (
          <div className="max-w-[84%] bg-gray-100 text-gray-500 rounded-2xl px-3 py-2 text-sm">
            ⚡ EnerVision AI is thinking…
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* Input */}
      <div className="mt-3 flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder={`Type your question… (e.g., ${suggestions[hintIndex]})`}
          className="flex-1 rounded-full border px-3 py-2 outline-none text-sm"
        />
        <button
          onClick={send}
          disabled={loading}
          className="px-4 py-2 rounded-full bg-emerald-600 text-white text-sm hover:bg-emerald-700 disabled:opacity-60"
        >
          Send
        </button>
      </div>
    </div>
  );
}
