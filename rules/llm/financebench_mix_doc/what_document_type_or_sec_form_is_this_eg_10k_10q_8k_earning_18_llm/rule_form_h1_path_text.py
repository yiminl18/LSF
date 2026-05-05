def rule_form_h1_path_text(doc: dict) -> list[dict]:
    """Match H1-like spans whose path_text itself is a form name."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").strip()
            level = ((span.get("structure") or {}).get("level") or "")
            if level == "H1" and re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
