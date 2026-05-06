def rule_page1_state_or_ein_in_same_path(doc: dict) -> list[dict]:
    """Match page-1 spans whose path_text itself is a state or EIN value block."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}).get("path_text") or "").strip()
            if span.get("page_no") == 1 and path:
                last = path.split("|")[-1].strip()
                if re.fullmatch(r"\d{2}-\d{7}", last) or (1 <= len(last.split()) <= 4 and not re.search(r"\d", last)):
                    out.append(span)
        return out
    except Exception:
        return []
