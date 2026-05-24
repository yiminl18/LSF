def rule_page1_8k_company_header_with_symbol_exchange(doc: dict) -> list[dict]:
    """Match company H1 cover spans on page 1 whose text_span includes symbol and exchange."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                txt = ((span.get("text_span") or "") + " " + (span.get("text") or "")).lower()
                if (
                    "trading symbol" in txt
                    and ("nasdaq" in txt or "stock exchange" in txt)
                ):
                    out.append(span)
        return out
    except Exception:
        return []
