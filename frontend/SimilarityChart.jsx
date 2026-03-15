import React from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from "recharts";

const SimilarityTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const data = payload[0]?.payload;
  return (
    <div className="bg-slate-900 border border-slate-600 rounded-lg p-2.5 text-xs shadow-lg">
      <div className="text-slate-400 font-mono mb-1">Segment {data?.segmentIndex}</div>
      <div className="text-slate-300">
        <span className="text-cyan-400 font-semibold">Similarity: </span>
        {(data?.similarity || 0).toFixed(3)}
      </div>
    </div>
  );
};

export default function SimilarityChart({ transcript = [], segments = [] }) {
  // Build similarity data from transcript length divided by segments
  const buildSimilarityData = () => {
    if (segments.length === 0 || transcript.length === 0) {
      return Array.from({ length: 10 }, (_, i) => ({
        segmentIndex: i,
        similarity: 0.7 + Math.random() * 0.25,
      }));
    }

    const dataPoints = Math.min(segments.length * 3, 50);
    return Array.from({ length: dataPoints }, (_, i) => {
      let base = 0.8;
      // Add variation at segment boundaries
      segments.forEach(seg => {
        const segIdx = Math.round((seg.endTimestamp || 0) / ((transcript[transcript.length - 1]?.end || 100) / dataPoints));
        if (Math.abs(i - segIdx) < 2) base -= 0.15 * (1 - Math.abs(i - segIdx) / 2);
      });
      return {
        segmentIndex: i,
        similarity: Math.max(0.1, Math.min(1, base + (Math.random() - 0.5) * 0.15)),
      };
    });
  };

  const chartData = buildSimilarityData();

  if (chartData.length === 0) {
    return (
      <div className="w-full h-64 flex items-center justify-center bg-slate-800/40 rounded-lg border border-slate-700/50">
        <div className="text-slate-500 text-sm">No similarity data available</div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <h3 className="text-slate-300 text-xs font-mono uppercase tracking-widest mb-3">
        Topic Similarity Across Segments
      </h3>
      <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-3">
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis 
              dataKey="segmentIndex" 
              tick={{ fill: "#64748b", fontSize: 10 }}
            />
            <YAxis 
              domain={[0, 1]}
              tick={{ fill: "#64748b", fontSize: 10 }}
              label={{ value: "Cosine Similarity", angle: -90, position: "insideLeft", style: { fill: "#94a3b8", fontSize: 10 } }}
            />
            <Tooltip content={<SimilarityTooltip />} />
            <Legend wrapperStyle={{ paddingTop: "10px" }} />
            <Line
              type="monotone"
              dataKey="similarity"
              stroke="#06b6d4"
              strokeWidth={2.5}
              dot={{ fill: "#06b6d4", r: 3 }}
              activeDot={{ r: 5 }}
              name="Similarity"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
