def rule_annual_report_phrase(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        spans: list[dict] = []
        seen: set[tuple] = set()

        def add_snippet(start_idx: int, end_idx: int) -> None:
            start_idx = max(0, start_idx)
            end_idx = min(len(lines), end_idx)
            if start_idx >= end_idx:
                return
            text = "\n".join((lines[i].get("text") or "").rstrip() for i in range(start_idx, end_idx)).strip()
            if not text:
                return
            first = lines[start_idx]
            span = {
                "page_no": first.get("page_no"),
                "line_no": first.get("line_no"),
                "line_no_end": lines[end_idx - 1].get("line_no"),
                "text": text,
            }
            key = (span.get("page_no"), span.get("line_no"), span.get("line_no_end"), span["text"])
            if key not in seen:
                seen.add(key)
                spans.append(span)

        for idx, line in enumerate(lines[:250]):
            raw = (line.get("text") or "").strip()
            if not raw:
                continue
            upper = " ".join(raw.split()).upper()
            if "ANNUAL REPORT ON FORM 10-K" in upper:
                add_snippet(idx, idx + 2)
                break
            if "ANNUAL REPORT TO SHAREHOLDERS" in upper and "FORM 10-K" in upper:
                add_snippet(idx, idx + 2)
                break

        return spans
    except Exception:
        return []

