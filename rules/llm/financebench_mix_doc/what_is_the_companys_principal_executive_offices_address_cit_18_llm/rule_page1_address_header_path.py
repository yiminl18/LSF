def rule_page1_address_header_path(doc: dict) -> list[dict]:
    """Match spans whose path_text itself is a street address under the company block on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            path = span.get("structure", {}).get("path_text", "")
            if re.search(r"\|\s*\d{1,5}\s", path) or "warmley, bristol" in path.lower():
                out.append(span)
        return out
    except Exception:
        return []
