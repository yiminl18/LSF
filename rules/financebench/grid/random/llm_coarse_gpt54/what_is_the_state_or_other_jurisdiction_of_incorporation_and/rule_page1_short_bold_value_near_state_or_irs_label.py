def rule_page1_short_bold_value_near_state_or_irs_label(doc: dict) -> list[dict]:
    """Match short bold value spans on page 1 adjacent to state or IRS label spans."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text", "") or "").strip()
            if span.get("page_no") != 1 or span.get("bold") != 1 or len(txt) > 40:
                continue
            window = " ".join((texts[j].get("text", "") or "") for j in range(max(0, i-2), min(len(texts), i+3)))
            if re.search(r"State or other jurisdiction|Employer Identification|I\.?R\.?S\.?", window, re.I):
                out.append(span)
        return out
    except Exception:
        return []
