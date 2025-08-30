// src/shared/ChatPanel.jsx
import React, { useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";

export default function ChatPanel({ role = "resident" }) {
  const [messages, setMessages] = useState([
    {
      from: "bot",
      text: `Hi! I'm EnerVision. I’ll use today’s data to answer your questions.\n\nTry: *"How can I cut my monthly bill by 20% based on today’s forecast?"*`,
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("⚡ Consulting live data…");

  const BASE_URL = "http://127.0.0.1:8000";

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Add user message
    const newMessages = [...messages, { from: "user", text: input }];
    setMessages(newMessages);
    setInput("");
    setLoading(true);
    setLoadingMsg("⚡ Consulting live data…");

    // Fallback if it takes too long
    const timeoutId = setTimeout(() => {
      setLoadingMsg("⌛ Still working… EnerVision AI is thinking deeply ⚡");
    }, 15000);

    try {
      const res = await axios.post(`${BASE_URL}/api/chat`, {
        prompt: input,
        role,
      });
      clearTimeout(timeoutId);
      setMessages([
        ...newMessages,
        { from: "bot", text: res.data.reply || "No reply generated." },
      ]);
    } catch (err) {
      clearTimeout(timeoutId);
      setMessages([
        ...newMessages,
        { from: "bot", text: "⚠️ Error contacting EnerVision AI." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-2xl shadow-lg">
      {/* Header */}
      <div className="px-4 py-3 border-b">
        <div className="font-semibold text-emerald-700">Ask EnerVision</div>
        <div className="text-xs text-gray-500">
          Role: {role} • Answers grounded in today’s data
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 text-sm">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`p-3 rounded-lg ${
              msg.from === "user"
                ? "bg-emerald-600 text-white self-end ml-10"
                : "bg-gray-50 text-gray-800 mr-10"
            }`}
          >
            <div className="prose prose-sm max-w-none">
              <ReactMarkdown>{msg.text}</ReactMarkdown>
            </div>
          </div>
        ))}

        {loading && (
          <div className="p-3 rounded-lg bg-gray-50 text-gray-500 text-sm italic flex items-center gap-2">
            <span className="animate-pulse">{loadingMsg}</span>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t p-3 flex gap-2">
        <input
          type="text"
          className="flex-grow border rounded-lg p-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
          placeholder="Type your question… (e.g., How can I save during peak hours?)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button
          onClick={sendMessage}
          disabled={loading}
          className="bg-emerald-600 text-white px-4 rounded-lg hover:bg-emerald-700 disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </div>
  );
}
