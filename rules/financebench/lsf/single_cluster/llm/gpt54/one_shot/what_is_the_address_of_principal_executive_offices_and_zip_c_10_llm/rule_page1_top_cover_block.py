def rule_page1_top_cover_block(doc: dict) -> list[dict]:
    """Return page-1 spans in the top cover block where registrant identity and address usually appear."""
    try:
        spans = doc.get("texts", [])
        out = []
        count = 0
        for span in spans:
            if span.get("page_no") == 1:
                out.append(span)
                count += 1
                if count >= 35:
                    break
        return out
    except Exception:
        return []
