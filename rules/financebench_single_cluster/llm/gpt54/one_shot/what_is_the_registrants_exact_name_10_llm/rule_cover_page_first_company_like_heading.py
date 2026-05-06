def rule_cover_page_first_company_like_heading(doc: dict) -> list[dict]:
    """Match the first page-1 bold heading after FORM 10-K that is not a form/SEC boilerplate heading."""
    try:
        texts = doc.get("texts", [])
        out = []
        seen_form = False
        for span in texts:
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if span.get("page_no") != 1:
                continue
            if "form 10-k" in low:
                seen_form = True
                continue
            if not seen_form:
                continue
            if span.get("bold") != 1:
                continue
            if span.get("label") not in {"section_header", "text"}:
                continue
            if low in {
                "or",
                "documents incorporated by reference",
                "part i",
            }:
                continue
            if "annual report pursuant" in low or "transition report pursuant" in low:
                continue
            if "commission file number" in low:
                continue
            if "exact name of registrant" in low:
                continue
            if any(ch.isalpha() for ch in txt):
                out.append(span)
                break
        return out
    except Exception:
        return []
