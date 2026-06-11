def rule_page1_spans_before_address_label(doc: dict) -> list[dict]:
    """Match page-1 spans in the company block before the address label, where state and EIN usually appear."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and "Address of principal executive offices" in (span.get("text", "") or ""):
                for prev in texts[max(0, i-8):i]:
                    if prev.get("page_no") == 1:
                        out.append(prev)
                break
        return out
    except Exception:
        return []
