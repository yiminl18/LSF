def rule_page1_under_company_header(doc: dict) -> list[dict]:
    """Match page-1 body spans under the main registrant/company header."""
    try:
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1:
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").strip()
            if span.get("label") not in {"text", "section_header"}:
                continue
            if any(k in path for k in [
                "amazon.com, inc.",
                "the boeing company",
                "costco wholesale corporation",
                "amcor plc",
                "corning incorporated",
                "johnson & johnson",
                "lockheed martin corporation",
                "nike, inc.",
                "ebay inc.",
            ]):
                out.append(span)
            elif txt.isupper() and len(txt.split()) <= 6 and span.get("bold") == 1:
                out.append(span)
        return out
    except Exception:
        return []
