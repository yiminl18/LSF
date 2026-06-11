def rule_page1_h1_following_commission_file(doc: dict) -> list[dict]:
    """Match the first page-1 H1 after a commission file number mention."""
    try:
        texts = doc.get("texts", [])
        out = []
        seen = False
        for span in texts:
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if span.get("page_no") == 1 and "commission file" in txt:
                seen = True
                continue
            if (
                seen
                and span.get("page_no") == 1
                and span.get("structure", {}).get("level") == "H1"
                and "form 10-" not in (span.get("text") or "").lower()
            ):
                out.append(span)
                break
        return out
    except Exception:
        return []
