def rule_page1_first_non_form_h1(doc: dict) -> list[dict]:
    """Match the first page-1 H1 after commission/form headers that is not a generic filing title."""
    try:
        import re
        texts = doc.get("texts", [])
        bad = re.compile(r"(UNITED STATES|SECURITIES AND EXCHANGE COMMISSION|FORM 10-|FORM 8-K|CURRENT REPORT|WASHINGTON, D\.C\.)", re.I)
        candidates = []
        for span in texts:
            if span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H1":
                txt = (span.get("text") or "").strip()
                if txt and not bad.search(txt):
                    candidates.append(span)
        return candidates[:1]
    except Exception:
        return []
