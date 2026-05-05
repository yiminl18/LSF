def rule_page1_main_company_name_neighbors(doc: dict) -> list[dict]:
    """Match spans neighboring the main company-name header on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header" and span.get("structure", {}).get("level") == "H1":
                tsp = (span.get("text_span") or "").lower()
                if "exact name of registrant" in tsp:
                    for j in range(max(0, i - 1), min(len(texts), i + 8)):
                        if texts[j].get("page_no") == 1:
                            out.append(texts[j])
        return out
    except Exception:
        return []
