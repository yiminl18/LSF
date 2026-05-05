def rule_form_heading_top_of_doc(doc: dict) -> list[dict]:
    """Match SEC form headings appearing in the first 15 spans of the document."""
    import re
    try:
        out = []
        for span in doc.get("texts", [])[:15]:
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", (span.get("text") or "").strip(), re.I):
                out.append(span)
        return out
    except Exception:
        return []
