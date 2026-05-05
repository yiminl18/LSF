def rule_form_heading_all_caps(doc: dict) -> list[dict]:
    """Match all-caps form headings, which are common on SEC cover pages."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", text, re.I) and span.get("all_cap", 0) == 1:
                out.append(span)
        return out
    except Exception:
        return []
