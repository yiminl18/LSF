def rule_page1_section12b_block(doc: dict) -> list[dict]:
    """Match page 1 spans in the Section 12(b) securities registration block."""
    try:
        results = []
        in_block = False
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = span.get("text", "").lower()
            path = span.get("structure", {}).get("path_text", "").lower()

            if "12(b)" in text or "trading symbol" in text or "trading symbol" in path:
                in_block = True

            if in_block:
                results.append(span)
                if "12(g)" in text or ("indicate by check mark" in text and len(results) > 2):
                    break
        return results
    except Exception:
        return []
