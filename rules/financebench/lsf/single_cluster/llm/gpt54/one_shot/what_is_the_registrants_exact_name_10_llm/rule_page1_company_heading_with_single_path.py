def rule_page1_company_heading_with_single_path(doc: dict) -> list[dict]:
    """Match page-1 headings whose path_text equals their text, a common cover-page company-name pattern."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "").strip()
            path = (span.get("structure", {}).get("path_text") or "").strip()
            if span.get("page_no") != 1:
                continue
            if not txt or not path:
                continue
            if txt != path:
                continue
            low = txt.lower()
            if low in {"form 10-k", "or", "documents incorporated by reference", "part i"}:
                continue
            if "commission" in low:
                continue
            if any(ch.isalpha() for ch in txt):
                out.append(span)
        return out
    except Exception:
        return []
