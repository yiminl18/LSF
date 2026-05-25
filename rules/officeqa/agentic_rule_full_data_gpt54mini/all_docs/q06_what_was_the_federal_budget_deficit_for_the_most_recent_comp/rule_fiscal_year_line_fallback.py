def rule_fiscal_year_line_fallback(doc: dict) -> list[dict]:
    try:
        import re

        spans: list[dict] = []
        seen = set()
        lines = doc.get("lines") or []

        percent_re = re.compile(
            r"(?i)\b\d+(?:\.\d+)?\s*(?:%|percent|per cent)\b.*\b(?:gdp|gnp|nominal gdp|gross national product)\b|\b(?:gdp|gnp|nominal gdp|gross national product)\b.*\b\d+(?:\.\d+)?\s*(?:%|percent|per cent)\b"
        )

        def compact(s: str) -> str:
            return "".join(ch for ch in s.lower() if ch.isalnum())

        def line_text(idx: int) -> str:
            parts = []
            for j in range(idx, min(len(lines), idx + 3)):
                txt = (lines[j].get("text") or "").strip()
                if txt:
                    parts.append(txt)
            return " ".join(parts)

        for i, line in enumerate(lines):
            text = (line.get("text") or "").strip()
            low = text.lower()
            ctext = compact(text)
            window = line_text(i)
            window_low = window.lower()
            window_compact = compact(window)
            if not text and not window:
                continue
            if not percent_re.search(window) and not re.search(
                r"\b\d+(?:\.\d+)?(?:%|percent|percentof|percentinrelationto|per cent)\b.*\b(?:gdp|gnp|nominalgdp|grossnationalproduct)\b",
                window_compact,
            ):
                continue
            if not any(term in window_low or term in window_compact for term in ("fiscal", "fy", "budget", "deficit", "surplus", "receipts", "outlays")):
                continue
            key = (
                line.get("page_no"),
                line.get("line_no"),
                window,
            )
            if key in seen:
                continue
            seen.add(key)
            spans.append(
                {
                    "page_no": line.get("page_no"),
                    "line_no": line.get("line_no"),
                    "text": window,
                }
            )

        return spans
    except Exception:
        return []
