import re
from typing import Any, Dict, List, Optional
import numpy as np

from engine.nlp_math import extract_top_keyphrases, tfidf_similarity_matrix

POSITIVE_WORDS = {
    "great", "awesome", "amazing", "excited", "happy", "love", "wonderful", "excellent",
    "good", "fantastic", "promoted", "proud", "thrilled", "inspired", "grateful", "win",
    "success", "brilliant", "helpful", "optimistic", "joy", "progress", "breakthrough",
}

NEGATIVE_WORDS = {
    "stressed", "overwhelmed", "anxious", "burned", "burnout", "sad", "tired", "exhausted",
    "stuck", "frustrated", "worried", "hard", "tough", "difficult", "upset", "afraid",
    "nervous", "failing", "lost", "depressed", "angry", "pain", "struggling",
}


class TextIntelligenceEngine:
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        words = re.findall(r"[a-z']+", text.lower())
        if not words:
            return {"score": 0.0, "label": "neutral"}
        pos = sum(1 for w in words if w in POSITIVE_WORDS)
        neg = sum(1 for w in words if w in NEGATIVE_WORDS)
        raw = (pos - neg) / max(3.0, np.sqrt(len(words)))
        score = float(np.clip(raw, -1.0, 1.0))
        label = "positive" if score > 0.15 else ("stressed/negative" if score < -0.15 else "neutral")
        return {"score": round(score, 3), "label": label, "pos_hits": pos, "neg_hits": neg}

    def summarize_text(self, raw_input: str, max_sentences: int = 3) -> Dict[str, Any]:
        # Strip leading instruction ("Summarize this text: ...")
        body = re.sub(
            r"^(?:please\s+)?(?:summarize|give\s+me\s+a\s+(?:tl;?dr|summary)\s+of|extract\s+(?:the\s+)?key\s+(?:bullet\s+)?points\s+from|condense|analyze\s+the\s+sentiment\s+and\s+keywords\s+of)\s*(?:this\s+(?:article|text|passage|paragraph|notes)?\s*(?:for\s+me)?\s*[:\-]?\s*)",
            "",
            raw_input.strip(),
            flags=re.IGNORECASE,
        ).strip()

        if len(body) < 40:
            return {
                "ok": False,
                "reason": "Paste a paragraph, article, or meeting notes after 'Summarize:' and I will run TextRank graph centrality to extract the core takeaways and keyphrases.",
            }

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", body) if len(s.strip()) > 15]
        words = re.findall(r"[A-Za-z0-9\-']+", body)
        sentiment = self.analyze_sentiment(body)

        # Extract top keyphrases
        keyphrases = extract_top_keyphrases(body, top_k=7)

        if len(sentences) <= max_sentences:
            selected = sentences
        else:
            try:
                sim_matrix = tfidf_similarity_matrix(sentences)
                np.fill_diagonal(sim_matrix, 0.0)
                # Power iteration PageRank over sentence similarity graph
                row_sums = sim_matrix.sum(axis=1, keepdims=True)
                transition = np.divide(sim_matrix, row_sums, out=np.zeros_like(sim_matrix), where=row_sums != 0)
                scores = np.ones(len(sentences)) / len(sentences)
                damping = 0.85
                for _ in range(25):
                    scores = (1 - damping) / len(sentences) + damping * transition.T.dot(scores)
                # Slightly favor opening thesis sentence
                scores[0] *= 1.15
                top_idx = sorted(np.argsort(scores)[::-1][:max_sentences])
                selected = [sentences[i] for i in top_idx]
            except Exception:
                selected = sentences[:max_sentences]

        compression = round((1.0 - (sum(len(s) for s in selected) / max(1, len(body)))) * 100, 1)
        return {
            "ok": True,
            "summary_points": selected,
            "keyphrases": keyphrases,
            "word_count": len(words),
            "sentence_count": len(sentences),
            "compression_pct": max(0.0, compression),
            "sentiment": sentiment,
        }

    def parse_chart_request(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extracts label-value pairs from natural prompts such as:
        'chart jan 120 feb 150 mar 210 apr 195' or 'python: 48, typescript: 36, rust: 22'
        """
        cleaned = re.sub(
            r"^(?:please\s+)?(?:chart|plot|graph|visualize|make\s+a\s+(?:bar|line|area)\s+(?:chart|graph)\s+(?:of|for)|create\s+a\s+chart\s+(?:comparing|of|for)|analyze\s+and\s+chart)\s+(?:this\s+data|these\s+metrics|these\s+numbers|sales\s+numbers)?\s*[:\-]?\s*",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        )
        pairs = re.findall(r"([A-Za-z][A-Za-z0-9_\-/]{0,18})\s*[:=]?\s*([\-+]?\d+(?:\.\d+)?)", cleaned)
        filtered = [
            (lbl, float(val))
            for lbl, val in pairs
            if lbl.lower() not in {"chart", "plot", "bar", "line", "area", "top", "last", "first", "over", "under"}
        ]
        if len(filtered) < 2:
            return None

        chart_type = "line" if any(k in text.lower() for k in ("line", "trend", "temperature", "over time", "trajectory")) else (
            "area" if "area" in text.lower() else "bar"
        )
        vals = [v for _, v in filtered]
        max_item = max(filtered, key=lambda x: x[1])
        min_item = min(filtered, key=lambda x: x[1])
        mean_val = float(np.mean(vals))

        return {
            "chart_artifact": {
                "type": "chart",
                "chart_type": chart_type,
                "title": "Interactive Data Analysis",
                "data": [{"label": lbl.capitalize(), "value": round(v, 3)} for lbl, v in filtered[:24]],
            },
            "stats": {
                "count": len(filtered),
                "mean": round(mean_val, 2),
                "peak": f"{max_item[0].capitalize()} ({max_item[1]:g})",
                "lowest": f"{min_item[0].capitalize()} ({min_item[1]:g})",
                "net_change_pct": round(((vals[-1] - vals[0]) / abs(vals[0])) * 100, 1) if vals[0] != 0 else 0.0,
            },
        }

    def solve_syllogism(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Solves universal/particular deductive syllogisms such as:
        'If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?'
        """
        m = re.search(
            r"all\s+([a-z]+)\s+are\s+([a-z]+)\s+and\s+all\s+([a-z]+)\s+are\s+([a-z]+).*?are\s+all\s+([a-z]+)\s+([a-z]+)",
            text.lower(),
        )
        if not m:
            return None
        a1, b1, b2, c1, q_sub, q_pred = m.groups()
        if b1 == b2 and q_sub == a1 and q_pred == c1:
            return {
                "valid": True,
                "conclusion": f"**Yes — deductively guaranteed (`TRUE`).** Every `{a1}` is necessarily a `{c1}`.",
                "proof": [
                    f"**Premise 1**: $\\forall x,\\; \\text{{{a1}}}(x) \\implies \\text{{{b1}}}(x)$ (`{a1} ⊆ {b1}`)",
                    f"**Premise 2**: $\\forall x,\\; \\text{{{b2}}}(x) \\implies \\text{{{c1}}}(x)$ (`{b2} ⊆ {c1}`)",
                    f"**Transitive Deduction (Barbara Syllogism)**: By set transitivity, `{a1} ⊆ {b1} ⊆ {c1}`, therefore $\\forall x,\\; \\text{{{a1}}}(x) \\implies \\text{{{c1}}}(x)$.",
                ],
            }
        if b1 == b2 and q_sub == c1 and q_pred == a1:
            return {
                "valid": False,
                "conclusion": f"**Not necessarily (`INDETERMINATE / Fallacy of Affirming the Consequent`).** All `{a1}` are `{c1}`, but there may exist `{c1}` outside `{a1}`.",
                "proof": [
                    f"**Known**: `{a1} ⊆ {b1} ⊆ {c1}`",
                    f"**Converse Check**: `{c1} ⊆ {a1}` does **not** follow unless the sets are proven equal (`{a1} = {c1}`).",
                ],
            }
        return None
