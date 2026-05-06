def rule_path_text_is_company_name(doc: dict) -> list[dict]:
    """Match H1 spans on page 1 whose path_text equals the span text and look like a company name."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        bad = re.compile(r"^(FORM 10-|FORM 8-K|CURRENT REPORT|SECURITIES AND EXCHANGE COMMISSION|UNITED STATES|WASHINGTON, D\.C\.)", re.I)
        for span in texts:
            txt = (span.get("text") or "").strip()
            path = (span.get("structure", {}).get("path_text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and txt
                and path == txt
                and not bad.search(txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
