def rule_page1_h1_after_form_header(doc: dict) -> list[dict]:
    """Match page-1 H1 section headers after FORM 10-K/10-Q/8-K that look like the company name."""
    try:
        texts = doc.get("texts", [])
        out = []
        seen_form = False
        for span in texts:
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and "form 10-" in txt.lower():
                seen_form = True
            if (
                seen_form
                and span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and txt
                and "form 10-" not in txt.lower()
                and "securities and exchange commission" not in txt.lower()
                and len(txt) < 120
            ):
                out.append(span)
        return out
    except Exception:
        return []
