def rule_cover_labels_state_ein(doc: dict) -> list[dict]:
    """Retrieve page-1 cover spans with state/EIN labels and adjacent value-like section headers."""
    try:
        import re
        out = []
        for s in doc.get("texts", []):
            if s.get("page_no") != 1:
                continue
            txt = (s.get("text") or "")
            path = ((s.get("structure") or {}).get("path_text") or "")
            label = s.get("label") or ""
            low = txt.lower()
            if "state or other jurisdiction of incorporation" in low or "state or other jurisdiction of incorporation or organization" in low:
                out.append(s)
                continue
            if "i.r.s. employer identification no" in low or "irs employer identification no" in low:
                out.append(s)
                continue
            if label == "section_header" and path:
                if re.search(r"\b\d{2}-\d{7}\b", txt) or re.search(r"\b(delaware|washington|new york|jersey)\b", low):
                    out.append(s)
        return out
    except Exception:
        return []

