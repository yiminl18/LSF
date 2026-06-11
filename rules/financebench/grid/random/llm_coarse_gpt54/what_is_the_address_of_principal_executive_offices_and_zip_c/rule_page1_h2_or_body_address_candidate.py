def rule_page1_h2_or_body_address_candidate(doc: dict) -> list[dict]:
    """Match page-1 H2/body spans that are likely the address line near the registrant block."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            level = ((span.get("structure") or {}).get("level") or "")
            if level not in {"H2", "Body"}:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'^\d{2,} ', txt) and len(txt) < 120:
                out.append(span)
        return out
    except Exception:
        return []
