def rule_form_in_path_text(doc: dict) -> list[dict]:
    """Match spans whose structure path_text itself is a form heading like FORM 10-K/10-Q/8-K."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}).get("path_text") or "").strip()
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K|20-F|6-K|S-1|S-3|S-4)\b", path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
