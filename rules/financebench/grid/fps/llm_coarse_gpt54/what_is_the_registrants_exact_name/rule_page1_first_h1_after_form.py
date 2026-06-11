def rule_page1_first_h1_after_form(doc: dict) -> list[dict]:
    """Match the first H1 on page 1 after the FORM heading that is not a form/report heading."""
    try:
        texts = doc.get("texts", [])
        out = []
        after_form = False
        for span in texts:
            txt = (span.get("text") or "").strip().lower()
            if span.get("page_no") != 1:
                continue
            if txt in {"form 10-k", "form 10-q", "form 8-k"}:
                after_form = True
                continue
            if not after_form:
                continue
            if span.get("structure", {}).get("level") == "H1" and span.get("bold") == 1:
                if "current report" not in txt and "annual report" not in txt and "quarterly report" not in txt and "or" != txt:
                    out.append(span)
                    break
        return out
    except Exception:
        return []
