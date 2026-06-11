def rule_page1_h1_before_part_i(doc: dict) -> list[dict]:
    """Match prominent page-1 company headings before any PART I content begins."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip().lower()
            if "part i" in txt:
                break
            if span.get("bold") == 1 and float(span.get("size") or 0) >= 10:
                if "form 10-" not in txt and "form 8-k" not in txt and "securities and exchange commission" not in txt and "current report" not in txt:
                    out.append(span)
        return out
    except Exception:
        return []
