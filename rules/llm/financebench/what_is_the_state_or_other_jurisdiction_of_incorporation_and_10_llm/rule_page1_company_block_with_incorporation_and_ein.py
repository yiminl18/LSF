def rule_page1_company_block_with_incorporation_and_ein(doc: dict) -> list[dict]:
    """Match page-1 company cover blocks whose combined text contains both incorporation and EIN cues."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            combined = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"exact name of registrant", combined, re.I):
                if re.search(r"(state|jurisdiction).{0,80}incorporation|state of incorporation", combined, re.I) and re.search(r"\d{2}-\d{7}|employer identification", combined, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
