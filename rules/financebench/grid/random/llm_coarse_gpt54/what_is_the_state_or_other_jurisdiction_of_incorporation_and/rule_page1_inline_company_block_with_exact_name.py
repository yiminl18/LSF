def rule_page1_inline_company_block_with_exact_name(doc: dict) -> list[dict]:
    """Match any page-1 span containing the exact-name caption, which often embeds the answer inline."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if span.get("page_no") == 1 and "Exact name of registrant as specified in its charter" in text:
                out.append(span)
        return out
    except Exception:
        return []
