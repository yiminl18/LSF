def rule_page1_h1_h2_phone_cluster(doc: dict) -> list[dict]:
    """Match phone-like spans on page 1 that are H1/H2 or directly under H1/H2 cover sections."""
    try:
        import re
        out = []
        phone_pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            level = ((span.get("structure") or {}).get("level") or "")
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if phone_pat.search(txt) and level in {"H1", "H2", "Body"}:
                path = ((span.get("structure") or {}).get("path_text") or "")
                if path:
                    out.append(span)
        return out
    except Exception:
        return []
