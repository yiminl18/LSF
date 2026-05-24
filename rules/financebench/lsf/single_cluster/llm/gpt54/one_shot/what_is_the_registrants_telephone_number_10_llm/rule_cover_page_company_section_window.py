def rule_cover_page_company_section_window(doc: dict) -> list[dict]:
    """Match a window of spans under the first company-name section on page 1, where the phone usually appears."""
    try:
        texts = doc.get("texts", [])
        start = None
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            path = ((span.get("structure", {}) or {}).get("path_text") or "").strip()
            level = ((span.get("structure", {}) or {}).get("level") or "")
            if span.get("label") == "section_header" and level in {"H1", "H2"}:
                txt = (span.get("text") or "").strip()
                if txt and "form 10-k" not in txt.lower() and "commission" not in txt.lower():
                    start = i
                    break
        if start is None:
            return []
        out = []
        for j in range(start, min(len(texts), start + 20)):
            if texts[j].get("page_no") == 1:
                out.append(texts[j])
        return out
    except Exception:
        return []
