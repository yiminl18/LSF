def rule_page1_combined_text_or_textspan_state_and_ein(doc: dict) -> list[dict]:
    """Match page-1 spans whose combined text/text_span contains both a state/jurisdiction phrase and EIN number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            combined = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"\d{2}-\d{7}", combined):
                if re.search(r"(state|jurisdiction|incorporation|organization)", combined, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
