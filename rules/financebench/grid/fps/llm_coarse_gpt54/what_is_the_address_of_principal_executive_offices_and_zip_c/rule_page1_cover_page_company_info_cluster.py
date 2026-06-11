def rule_page1_cover_page_company_info_cluster(doc: dict) -> list[dict]:
    """Match spans in the dense page-1 company information cluster that contain the principal office address."""
    try:
        import re
        texts = [s for s in doc.get("texts", []) if s.get("page_no") == 1]
        out = []
        for i, span in enumerate(texts):
            full = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'exact name of registrant', full, re.I):
                cluster = texts[i:i+15]
                for s in cluster:
                    t = (s.get("text") or "").strip()
                    if re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I):
                        out.append(s)
                break
        return out
    except Exception:
        return []
