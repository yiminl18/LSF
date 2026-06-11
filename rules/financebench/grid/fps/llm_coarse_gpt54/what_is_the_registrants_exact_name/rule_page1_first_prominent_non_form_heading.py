def rule_page1_first_prominent_non_form_heading(doc: dict) -> list[dict]:
    """Match the first prominent page-1 heading after FORM/SEC headers."""
    try:
        texts = doc.get("texts", [])
        out = []
        bad = {"form 10-k", "form 10-q", "form 8-k", "current report", "or"}
        for span in texts:
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if span.get("page_no") != 1:
                continue
            if not txt:
                continue
            if "securities and exchange commission" in low or "washington, d.c." in low or "washington, dc" in low:
                continue
            if low in bad:
                continue
            if span.get("bold") == 1 and float(span.get("size") or 0) >= 12 and span.get("label") in {"section_header", "text"}:
                out.append(span)
                break
        return out
    except Exception:
        return []
