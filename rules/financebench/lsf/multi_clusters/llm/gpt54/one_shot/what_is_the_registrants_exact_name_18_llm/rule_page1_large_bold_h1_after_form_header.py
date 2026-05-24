def rule_page1_large_bold_h1_after_form_header(doc: dict) -> list[dict]:
    """Match large bold H1 company-name headers on page 1 appearing after FORM 10-K/10-Q/8-K."""
    try:
        texts = doc.get("texts", [])
        seen_form = False
        out = []
        for span in texts:
            txt = (span.get("text", "") or "").strip()
            if span.get("page_no") != 1:
                continue
            if "form 10-" in txt.lower() or txt.lower() == "form 8-k":
                seen_form = True
            if (
                seen_form
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and span.get("bold") == 1
                and float(span.get("size", 0) or 0) >= 10
                and "exact name of registrant" not in txt.lower()
                and "form 10-" not in txt.lower()
                and "form 8-k" not in txt.lower()
                and "securities and exchange commission" not in txt.lower()
                and "current report" not in txt.lower()
            ):
                out.append(span)
        return out
    except Exception:
        return []
