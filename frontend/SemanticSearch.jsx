import React, { useState, useRef } from "react";

const API_BASE_URL = "http://localhost:5000";

export default function SemanticSearch({ segments = [], onHighlightSegment }) {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState("");
  const streamRef = useRef(null);

  const ask = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setResponse(null);
    setStreaming("");

    try {
      // Denormalize segments back to snake_case for backend
      const denormSegments = segments.map(seg => ({
        id: seg.id,
        content: seg.content,
        start_timestamp: seg.startTimestamp,
        end_timestamp: seg.endTimestamp,
        sentiment: seg.sentiment,
        keywords: seg.keywords,
      }));

      const resp = await fetch(`${API_BASE_URL}/api/rag`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, segments: denormSegments }),
      });

      if (!resp.ok) throw new Error(`Request failed: ${resp.status}`);
      const result = await resp.json();

      setLoading(false);

      // Stream in the answer
      let i = 0;
      const text = result.answer || "";
      clearInterval(streamRef.current);
      streamRef.current = setInterval(() => {
        i += 3;
        setStreaming(text.slice(0, i));
        if (i >= text.length) {
          clearInterval(streamRef.current);
          setResponse(result);
          setStreaming("");
        }
      }, 18);
    } catch (err) {
      setLoading(false);
      setResponse({
        answer: `Error: ${err.message}. Make sure the backend is running on http://localhost:5000`,
        referenced_segments: [],
        model: "error",
        context_length: 0,
      });
    }
  };

  const suggestions = ["What are the main topics?", "What is the overall sentiment?", "Summarize key points"];

  return (
    <div className="w-full">
      <h3 className="text-slate-300 text-xs font-mono uppercase tracking-widest mb-4 flex items-center gap-2">
        <span>🤖 Semantic Search</span>
        <span className="text-slate-600 text-xs">RAG Q&A</span>
      </h3>

      <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4 space-y-3">
        {/* Search input */}
        <div className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
            placeholder="Ask a question about the content..."
            className="flex-1 bg-slate-900/60 border border-slate-600 rounded-lg px-4 py-2.5 text-slate-200 text-sm placeholder:text-slate-600 focus:outline-none focus:border-indigo-500 transition-colors"
          />
          <button
            onClick={ask}
            disabled={loading}
            className="px-4 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-sm font-medium transition-all"
          >
            {loading ? "..." : "Ask"}
          </button>
        </div>

        {/* Suggestions */}
        <div className="flex flex-wrap gap-2">
          {suggestions.map((s) => (
            <button
              key={s}
              onClick={() => {
                setQuery(s);
              }}
              className="text-xs text-slate-400 border border-slate-700 rounded-lg px-3 py-1 hover:border-slate-500 hover:text-slate-200 transition-all"
            >
              {s}
            </button>
          ))}
        </div>

        {/* Response */}
        {(streaming || response) && (
          <div className="mt-4 pt-4 border-t border-slate-700/50 space-y-3">
            <div className="bg-slate-900/60 rounded-lg p-4 border border-slate-700/30">
              <div className="text-sm text-slate-300 leading-relaxed">
                {streaming || response?.answer}
                {streaming && <span className="inline-block w-0.5 h-4 bg-cyan-400 ml-0.5 animate-pulse" />}
              </div>
            </div>

            {/* Referenced segments */}
            {response && response.referenced_segments && response.referenced_segments.length > 0 && (
              <div className="space-y-2">
                <div className="text-xs text-slate-500 font-mono">Referenced Segments:</div>
                <div className="flex flex-wrap gap-2">
                  {response.referenced_segments.map((segId) => (
                    <button
                      key={segId}
                      onClick={() => onHighlightSegment?.(segId)}
                      className="px-3 py-1.5 rounded-lg border border-indigo-500/50 bg-indigo-500/10 text-indigo-300 text-xs hover:bg-indigo-500/20 transition-all"
                    >
                      Segment {segId}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Model info */}
            {response?.model && response.model !== "error" && (
              <div className="text-xs text-slate-500 flex items-center justify-between">
                <span>Model:</span>
                <span className="text-cyan-400 font-mono">{response.model}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
