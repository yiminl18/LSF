def rule_page1_h1_with_company_suffix(doc: dict) -> list[dict]:
    """Match page-1 H1 spans ending with common company suffixes like Inc., Corporation, Company, plc."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        suffix = re.compile(r"\b(inc\.?|corporation|company|plc|ltd\.?|limited)\b", re.I)
        for span in texts:
            txt = (span.get("text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("structure", {}).get("level") == "H1"
                and suffix.search(txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
