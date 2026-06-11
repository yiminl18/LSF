def rule_page1_3m_style_company_block(doc: dict) -> list[dict]:
    """Match page-1 company H1 and nearby H2/body spans in split-value 10-Q/8-K cover pages."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header" and span.get("structure", {}).get("level") == "H1":
                txt = span.get("text", "") or ""
                if "COMPANY" in txt.upper() or "INC." in txt.upper() or "PLC" in txt.upper():
                    for cand in texts[i:i+20]:
                        if cand.get("page_no") == 1:
                            out.append(cand)
                    break
        return out
    except Exception:
        return []
