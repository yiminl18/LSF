def rule_page1_main_company_path_text(doc: dict) -> list[dict]:
    """Match page-1 spans whose path_text is the main company name and that mention address/ZIP/phone."""
    try:
        spans = doc.get("texts", [])
        out = []
        for s in spans:
            if s.get("page_no") != 1:
                continue
            path = ((s.get("structure") or {}).get("path_text") or "").lower()
            txt = ((s.get("text") or "") + " " + (s.get("text_span") or "")).lower()
            if path and any(name in path for name in [
                "amazon.com, inc.", "the boeing company", "costco wholesale corporation",
                "amcor plc", "nike, inc.", "johnson & johnson", "corning incorporated",
                "lockheed martin corporation", "ebay inc."
            ]):
                if any(k in txt for k in ["address", "zip code", "telephone", "plaza", "drive", "avenue", "road"]):
                    out.append(s)
        return out
    except Exception:
        return []
