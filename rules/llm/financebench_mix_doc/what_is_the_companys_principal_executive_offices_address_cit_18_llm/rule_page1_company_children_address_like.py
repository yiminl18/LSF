def rule_page1_company_children_address_like(doc: dict) -> list[dict]:
    """Match child body spans under the main company header on page 1 that look like address lines."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        company_paths = set()
        for span in texts:
            if span.get("page_no") == 1 and span.get("label") == "section_header" and span.get("structure", {}).get("level") == "H1":
                tsp = (span.get("text_span") or "").lower()
                if "exact name of registrant" in tsp:
                    company_paths.add(span.get("structure", {}).get("path_text"))
        for span in texts:
            path = span.get("structure", {}).get("path_text", "")
            txt = span.get("text") or ""
            if span.get("page_no") != 1:
                continue
            if path in company_paths and span.get("structure", {}).get("level") == "Body":
                if re.search(r"\b[A-Z][a-zA-Z\.\- ]+,\s*(?:[A-Z]{2}|[A-Z][a-z]+(?: [A-Z][a-z]+)*)", txt):
                    out.append(span)
                elif re.search(r"\b\d{3,5}\b", txt) and re.search(r"[A-Za-z]+,\s*[A-Za-z]{2,}", txt):
                    out.append(span)
        return out
    except Exception:
        return []
