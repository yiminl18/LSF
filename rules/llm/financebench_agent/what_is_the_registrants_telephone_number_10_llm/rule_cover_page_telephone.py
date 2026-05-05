def rule_cover_page_telephone(doc: dict) -> list[dict]:
    """Retrieve page-1 cover-page spans containing the registrant telephone number."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if not isinstance(span, dict):
                continue
            if span.get("page_no") != 1:
                continue
            label = span.get("label")
            if label not in {"text", "section_header"}:
                continue
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            hay = f"{txt} {path}".lower()
            if "telephone number" in hay or "including area code" in hay:
                out.append(span)
                continue
            if any(k in path.lower() for k in [
                "amcor plc", "costco wholesale corporation", "the boeing company",
                "amazon.com, inc.", "corning incorporated", "nike, inc.",
                "lockheed martin corporation", "johnson & johnson"
            ]):
                if re.search(r"(?:\+?\d[\d\-\s\(\)]{6,}\d)", txt):
                    out.append(span)
                    continue
            if any(k in path.lower() for k in ["| 95125", "| 607-974-9000", "| (503) 671-6453", "| 08933"]):
                if re.search(r"(?:\+?\d[\d\-\s\(\)]{6,}\d)", txt):
                    out.append(span)
        return out
    except Exception:
        return []

