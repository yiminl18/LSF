def rule_page1_company_header_near_commission_file_number(doc: dict) -> list[dict]:
    """Match page-1 company headers appearing within a few spans after a commission file number line."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = span.get("text") or ""
            if span.get("page_no") == 1 and re.search(r"Commission (file|File) number|Commission File No\.?|Commission File Number", txt, re.I):
                for j in range(i + 1, min(i + 8, len(texts))):
                    nxt = texts[j]
                    nt = (nxt.get("text") or "").strip()
                    if nxt.get("page_no") != 1:
                        break
                    if nxt.get("structure", {}).get("level") == "H1" and "FORM" not in nt and "CURRENT REPORT" not in nt:
                        out.append(nxt)
        return out
    except Exception:
        return []
