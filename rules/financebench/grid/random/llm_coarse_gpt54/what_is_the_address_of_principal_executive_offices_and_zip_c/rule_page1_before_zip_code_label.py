def rule_page1_before_zip_code_label(doc: dict) -> list[dict]:
    """Match the span immediately preceding a zip code label on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") == 1:
                txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
                if re.search(r'\bzip code\b', txt, re.I):
                    if i > 0 and texts[i - 1].get("page_no") == 1:
                        out.append(texts[i - 1])
        return out
    except Exception:
        return []
