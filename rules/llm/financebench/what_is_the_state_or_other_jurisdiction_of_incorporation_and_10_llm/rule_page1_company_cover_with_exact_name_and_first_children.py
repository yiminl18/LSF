def rule_page1_company_cover_with_exact_name_and_first_children(doc: dict) -> list[dict]:
    """Match the company cover header containing exact-name text and its immediate children."""
    try:
        texts = doc.get("texts", [])
        out = []
        for idx, span in enumerate(texts):
            combined = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if span.get("page_no") == 1 and "exact name of registrant" in combined:
                out.append(span)
                for child in texts:
                    if child.get("page_no") == 1 and child.get("structure", {}).get("parent_id") == idx:
                        out.append(child)
                break
        return out
    except Exception:
        return []
