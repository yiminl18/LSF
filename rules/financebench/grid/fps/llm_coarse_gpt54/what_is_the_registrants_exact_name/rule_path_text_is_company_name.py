def rule_path_text_is_company_name(doc: dict) -> list[dict]:
    """Match spans whose structure.path_text equals their own text and are prominent page-1 headings."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "").strip()
            path = (span.get("structure", {}).get("path_text") or "").strip()
            low = txt.lower()
            if span.get("page_no") != 1:
                continue
            if not txt or txt != path:
                continue
            if span.get("bold") != 1:
                continue
            if float(span.get("size") or 0) < 12:
                continue
            if low in {
                "form 10-k", "form 10-q", "form 8-k",
                "current report", "part i", "or"
            }:
                continue
            if "securities and exchange commission" in low:
                continue
            out.append(span)
        return out
    except Exception:
        return []
