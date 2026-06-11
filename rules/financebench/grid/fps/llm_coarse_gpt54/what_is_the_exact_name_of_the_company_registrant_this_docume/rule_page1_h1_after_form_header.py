def rule_page1_h1_after_form_header(doc: dict) -> list[dict]:
    """Match the first H1 on page 1 appearing after a FORM header and not itself a form/SEC heading."""
    try:
        texts = doc.get("texts", [])
        seen_form = False
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text", "") or "").strip()
            low = txt.lower()
            if "form 10-k" in low or "form 10-q" in low or "form 8-k" in low:
                seen_form = True
                continue
            if not seen_form:
                continue
            if span.get("label") == "section_header" and span.get("structure", {}).get("level") == "H1":
                if "securities and exchange commission" in low or "current report" in low or low == "or":
                    continue
                out.append(span)
                break
        return out
    except Exception:
        return []
