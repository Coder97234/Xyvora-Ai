import React, { useState } from "react";
import { Check, Copy, Play } from "lucide-react";

interface MarkdownViewProps {
  content: string;
  onRunPython?: (code: string) => void;
}

function renderInline(text: string): React.ReactNode[] {
  // Tokenize inline **bold**, *italic*, `code`, and ~~strikethrough~~
  const pattern = /(\*\*[^*]+\*\*|~~[^~]+~~|`[^`]+`|\*[^*\n]+\*)/g;
  const parts = text.split(pattern);
  return parts.map((part, idx) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return (
        <strong key={idx} className="font-semibold text-white">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("~~") && part.endsWith("~~") && part.length > 4) {
      return (
        <del key={idx} className="text-slate-500">
          {part.slice(2, -2)}
        </del>
      );
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return (
        <code
          key={idx}
          className="px-1.5 py-0.5 rounded bg-slate-900/95 border border-cyan-500/20 text-cyan-200 font-mono text-[12.5px]"
        >
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
      return (
        <em key={idx} className="italic text-slate-300">
          {part.slice(1, -1)}
        </em>
      );
    }
    return <React.Fragment key={idx}>{part}</React.Fragment>;
  });
}

const CodeFenceBlock: React.FC<{
  lang: string;
  code: string;
  onRunPython?: (code: string) => void;
}> = ({ lang, code, onRunPython }) => {
  const [copied, setCopied] = useState(false);
  const isPython = !lang || lang === "python" || lang === "py";

  const copyCode = () => {
    navigator.clipboard?.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  return (
    <div className="my-2.5 rounded-xl overflow-hidden border border-slate-800 bg-[#040812]">
      <div className="flex items-center justify-between px-3 py-1.5 bg-slate-900/90 border-b border-slate-800 text-[11px] font-mono text-slate-400">
        <span>{lang || "code"}</span>
        <div className="flex items-center gap-2">
          {isPython && lang === "python" && onRunPython && (
            <button
              type="button"
              onClick={() => onRunPython(code)}
              className="flex items-center gap-1 text-emerald-300 hover:text-emerald-200 transition"
            >
              <Play className="w-3 h-3" /> Run in Lab
            </button>
          )}
          <button
            type="button"
            onClick={copyCode}
            className="flex items-center gap-1 text-slate-300 hover:text-white transition"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      </div>
      <pre className="p-3 text-xs font-mono text-slate-200 overflow-x-auto leading-relaxed">
        {code}
      </pre>
    </div>
  );
};

export const MarkdownView: React.FC<MarkdownViewProps> = ({ content, onRunPython }) => {
  // Split fenced code blocks first
  const segments = content.split(/(```[\s\S]*?```)/g);

  return (
    <div className="space-y-2.5 text-[14px] leading-relaxed text-slate-200">
      {segments.map((seg, sIdx) => {
        if (seg.startsWith("```") && seg.endsWith("```")) {
          const firstNl = seg.indexOf("\n");
          const lang = firstNl > 3 ? seg.slice(3, firstNl).trim().toLowerCase() : "";
          const body = firstNl > -1 ? seg.slice(firstNl + 1, -3).replace(/\n$/, "") : seg.slice(3, -3);
          return <CodeFenceBlock key={sIdx} lang={lang} code={body} onRunPython={onRunPython} />;
        }

        const lines = seg.split("\n");
        const blocks: React.ReactNode[] = [];
        let i = 0;

        while (i < lines.length) {
          const line = lines[i];
          const trimmed = line.trim();

          if (!trimmed) {
            i++;
            continue;
          }

          // Horizontal rule
          if (trimmed === "---" || trimmed === "***") {
            blocks.push(<hr key={`${sIdx}-${i}`} className="border-slate-800 my-2" />);
            i++;
            continue;
          }

          // Headings
          if (trimmed.startsWith("### ")) {
            blocks.push(
              <h3 key={`${sIdx}-${i}`} className="text-base font-bold text-cyan-100 pt-1">
                {renderInline(trimmed.slice(4))}
              </h3>
            );
            i++;
            continue;
          }
          if (trimmed.startsWith("## ")) {
            blocks.push(
              <h2 key={`${sIdx}-${i}`} className="text-lg font-bold text-white pt-1">
                {renderInline(trimmed.slice(3))}
              </h2>
            );
            i++;
            continue;
          }

          // Markdown Table
          if (trimmed.startsWith("|") && i + 1 < lines.length && lines[i + 1].trim().startsWith("|")) {
            const tableLines: string[] = [];
            while (i < lines.length && lines[i].trim().startsWith("|")) {
              tableLines.push(lines[i].trim());
              i++;
            }
            const parseRow = (r: string) =>
              r
                .replace(/^\||\|$/g, "")
                .split("|")
                .map((c) => c.trim());
            const header = parseRow(tableLines[0]);
            const bodyRows = tableLines.slice(2).map(parseRow);
            blocks.push(
              <div key={`${sIdx}-tbl-${i}`} className="overflow-x-auto my-2 rounded-lg border border-slate-800">
                <table className="w-full text-xs text-left border-collapse">
                  <thead className="bg-slate-900/90 text-cyan-200">
                    <tr>
                      {header.map((h, hi) => (
                        <th key={hi} className="px-3 py-2 border-b border-slate-800 font-semibold">
                          {renderInline(h)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70 bg-slate-950/60">
                    {bodyRows.map((row, ri) => (
                      <tr key={ri}>
                        {row.map((cell, ci) => (
                          <td key={ci} className="px-3 py-2 text-slate-300">
                            {renderInline(cell)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
            continue;
          }

          // Blockquote
          if (trimmed.startsWith(">")) {
            const quoteLines: string[] = [];
            while (i < lines.length && lines[i].trim().startsWith(">")) {
              quoteLines.push(lines[i].trim().replace(/^>\s?/, ""));
              i++;
            }
            blocks.push(
              <blockquote
                key={`${sIdx}-q-${i}`}
                className="border-l-2 border-cyan-400/70 bg-cyan-950/20 pl-3.5 pr-3 py-2 rounded-r-lg text-slate-300 text-[13px] space-y-1"
              >
                {quoteLines.map((ql, qi) => (
                  <div key={qi}>{renderInline(ql)}</div>
                ))}
              </blockquote>
            );
            continue;
          }

          // Unordered or Ordered List
          if (/^[-*]\s+/.test(trimmed) || /^\d+\.\s+/.test(trimmed)) {
            const isOrdered = /^\d+\.\s+/.test(trimmed);
            const listItems: string[] = [];
            while (
              i < lines.length &&
              (/^[-*]\s+/.test(lines[i].trim()) || /^\d+\.\s+/.test(lines[i].trim()))
            ) {
              listItems.push(lines[i].trim().replace(/^([-*]|\d+\.)\s+/, ""));
              i++;
            }
            const ListTag = isOrdered ? "ol" : "ul";
            blocks.push(
              <ListTag
                key={`${sIdx}-lst-${i}`}
                className={`space-y-1.5 pl-5 ${isOrdered ? "list-decimal" : "list-disc"} marker:text-cyan-400`}
              >
                {listItems.map((li, liIdx) => (
                  <li key={liIdx} className="text-slate-200">
                    {renderInline(li)}
                  </li>
                ))}
              </ListTag>
            );
            continue;
          }

          // Standard Paragraph
          blocks.push(
            <p key={`${sIdx}-p-${i}`} className="text-slate-200">
              {renderInline(trimmed)}
            </p>
          );
          i++;
        }

        return <React.Fragment key={sIdx}>{blocks}</React.Fragment>;
      })}
    </div>
  );
};
