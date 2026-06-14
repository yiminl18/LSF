def rule_page1_date_after_split_email_header(doc: dict) -> list[dict]:
    """Match page-1 dates that follow a split VIA/email header and still precede Dear."""
    try:
        import re

        texts = doc.get("texts", [])
        date_re = re.compile(
            r"^(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\s*,?\s*\d{4}$",
            re.I,
        )
        via_re = re.compile(r"(?:via\s+electronic\s+mail|via\s+e-?mail)", re.I)

        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1 or span.get("label") != "text":
                continue
            raw = span.get("text") or ""
            norm = re.sub(r"(?<=\d)\s+(?=\d)", "", " ".join(raw.split()))
            if not date_re.match(norm):
                continue

            prev_spans = [s for s in texts[max(0, i - 3):i] if s.get("page_no") == 1]
            if not any(via_re.search((s.get("text") or "")) for s in prev_spans):
                continue
            if not any("@" in (s.get("text") or "") for s in prev_spans):
                continue

            next_spans = [s for s in texts[i + 1:i + 6] if s.get("page_no") == 1]
            if any(((s.get("text") or "").strip().lower().startswith("dear ")) for s in next_spans):
                out.append(span)
        return out
    except Exception:
        return []
