def rule_page1_company_block_first_20_children(doc: dict) -> list[dict]:
    """Match the first 20 child spans under the company H1 on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        company_parent = None
        for i, span in enumerate(texts):
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "FORM 10-" not in (span.get("text", "") or "")
                and "SECURITIES AND EXCHANGE COMMISSION" not in (span.get("text", "") or "")
            ):
                company_parent = span.get("structure", {}).get("parent_id", None)
                start = i
                for child in texts[i+1:i+21]:
                    if child.get("page_no") == 1:
                        out.append(child)
                break
        return out
    except Exception:
        return []
