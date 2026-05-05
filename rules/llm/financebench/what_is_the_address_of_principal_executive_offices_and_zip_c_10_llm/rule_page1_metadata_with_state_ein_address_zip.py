def rule_page1_metadata_with_state_ein_address_zip(doc: dict) -> list[dict]:
    """Match the classic SEC cover-page metadata sequence of state, EIN, address, and ZIP."""
    try:
        spans = doc.get("texts", [])
        out = []
        for i, span in enumerate(spans):
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "state or other jurisdiction" in txt or "state of incorporation" in txt:
                for j in range(i, min(len(spans), i + 10)):
                    if spans[j].get("page_no") == 1:
                        out.append(spans[j])
        return out
    except Exception:
        return []
