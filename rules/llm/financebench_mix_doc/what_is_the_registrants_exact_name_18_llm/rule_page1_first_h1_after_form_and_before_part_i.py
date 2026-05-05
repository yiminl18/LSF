def rule_page1_first_h1_after_form_and_before_part_i(doc: dict) -> list[dict]:
    """Match the first H1 on page 1 after a form header and before any PART I section."""
    try:
        texts = doc.get("texts", [])
        seen_form = False
        out = []
        for span in texts:
            txt = (span.get("text", "") or "").lower()
            if span.get("page_no") == 1 and ("form 10-" in txt or "form 8-k" in txt):
                seen_form = True
                continue
            if "part i" in txt and span.get("page_no", 0) >= 1:
                break
            if (
                seen_form
                and span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "securities and exchange commission" not in txt
                and "current report" not in txt
                and "form 10-" not in txt
                and "form 8-k" not in txt
            ):
                out.append(span)
                break
        return out
    except Exception:
        return []
