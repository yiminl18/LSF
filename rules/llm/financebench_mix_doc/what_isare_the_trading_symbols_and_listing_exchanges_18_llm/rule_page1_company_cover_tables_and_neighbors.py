def rule_page1_company_cover_tables_and_neighbors(doc: dict) -> list[dict]:
    """Match page-1 tables in the company cover block plus nearby spans."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("label") == "table" and span.get("page_no") == 1:
                cells = (((span.get("table_data") or {}).get("cells")) or [])
                cell_text = " ".join((c.get("text") or "").lower() for c in cells)
                if "trading symbol" in cell_text or "exchange on which registered" in cell_text:
                    for j in range(max(0, i - 2), min(len(texts), i + 3)):
                        if texts[j].get("page_no") == 1:
                            out.append(texts[j])
        return out
    except Exception:
        return []
