import React, { useMemo } from "react";

export default function KeywordExplorer({ segments = [], onKeywordClick, highlightedKeyword }) {
  // Extract and count all keywords
  const keywordFrequency = useMemo(() => {
    const freq = {};
    segments.forEach(seg => {
      (seg.keywords || []).forEach(kw => {
        freq[kw] = (freq[kw] || 0) + 1;
      });
    });
    return freq;
  }, [segments]);

  // Sort keywords by frequency
  const sortedKeywords = useMemo(
    () => Object.entries(keywordFrequency).sort((a, b) => b[1] - a[1]),
    [keywordFrequency]
  );

  // Determine font size based on frequency
  const maxFreq = Math.max(...Object.values(keywordFrequency), 1);
  const getSize = (freq) => {
    const ratio = freq / maxFreq;
    if (ratio > 0.7) return "text-lg";
    if (ratio > 0.4) return "text-base";
    return "text-sm";
  };

  const getColor = (freq) => {
    const ratio = freq / maxFreq;
    if (ratio > 0.7) return "border-indigo-500/60 bg-indigo-500/15 text-indigo-300 hover:bg-indigo-500/25";
    if (ratio > 0.4) return "border-cyan-500/50 bg-cyan-500/10 text-cyan-300 hover:bg-cyan-500/20";
    return "border-slate-600/50 bg-slate-700/30 text-slate-300 hover:bg-slate-700/50";
  };

  return (
    <div className="w-full">
      <h3 className="text-slate-300 text-xs font-mono uppercase tracking-widest mb-4">
        Keyword Explorer
      </h3>
      {sortedKeywords.length === 0 ? (
        <div className="text-slate-500 text-sm text-center py-8">
          No keywords found
        </div>
      ) : (
        <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-5">
          <div className="flex flex-wrap gap-3 justify-center sm:justify-start">
            {sortedKeywords.map(([keyword, freq]) => {
              const isHighlighted = keyword === highlightedKeyword;
              return (
                <button
                  key={keyword}
                  onClick={() => onKeywordClick?.(keyword)}
                  className={`
                    ${getSize(freq)} font-mono px-3 py-1.5 rounded-lg border transition-all duration-200
                    ${isHighlighted 
                      ? "border-yellow-500 bg-yellow-500/20 text-yellow-300 ring-2 ring-yellow-500/40" 
                      : getColor(freq)
                    }
                  `}
                  title={`Appears in ${freq} segment${freq > 1 ? 's' : ''}`}
                >
                  {keyword}
                </button>
              );
            })}
          </div>
          <div className="mt-4 text-xs text-slate-500 text-center">
            {sortedKeywords.length} unique keywords found
          </div>
        </div>
      )}
    </div>
  );
}
