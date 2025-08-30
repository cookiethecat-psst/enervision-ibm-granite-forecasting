// src/components/ChatMessage.jsx
import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";

export default function ChatMessage({ role = "assistant", text = "" }) {
  const clean = normalizeEnerVisionMarkdown(text);

  return (
    <div className={`mb-3 ${role === "user" ? "text-right" : ""}`}>
      <div
        className={`inline-block max-w-[70ch] rounded-2xl px-4 py-3 shadow-sm
        ${role === "user" ? "bg-blue-50" : "bg-gray-50"}`}
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeSanitize]}
          components={{
            // prevent heavy bold; render as plain text
            strong: ({ node, ...props }) => <span {...props} />,
            em: ({ node, ...props }) => <span className="italic" {...props} />,
            h3: ({ node, ...props }) => (
              <h3 className="text-base font-semibold mt-1 mb-2" {...props} />
            ),
            h4: ({ node, ...props }) => (
              <h4 className="text-sm font-semibold mt-1 mb-2" {...props} />
            ),
            ul: ({ node, ...props }) => (
              <ul className="list-disc pl-5 space-y-1" {...props} />
            ),
            ol: ({ node, ...props }) => (
              <ol className="list-decimal pl-5 space-y-1" {...props} />
            ),
            p: ({ node, ...props }) => (
              <p className="leading-relaxed mb-2" {...props} />
            ),
            code: ({ node, inline, ...props }) =>
              inline ? (
                <code className="px-1 py-0.5 rounded bg-black/5" {...props} />
              ) : (
                <code
                  className="block p-3 rounded bg-black/5 overflow-x-auto"
                  {...props}
                />
              ),
          }}
        >
          {clean}
        </ReactMarkdown>
      </div>
    </div>
  );
}

function normalizeEnerVisionMarkdown(md) {
  let s = md;

  // Lines like "**Heading:**"  → "#### Heading"
  s = s.replace(/^\s*\*\*(.+?)\:\*\*\s*$/gm, "#### $1");

  // Lines like "**1. Heading:**" → "1. Heading"
  s = s.replace(/^\s*\*\*(\d+\.\s+)(.+?)\:\*\*\s*$/gm, "$1$2");

  // Remove any remaining double-asterisk bold
  s = s.replace(/\*\*(.+?)\*\*/g, "$1");

  // Collapse extra blank lines
  s = s.replace(/\n{3,}/g, "\n\n");

  return s.trim();
}
