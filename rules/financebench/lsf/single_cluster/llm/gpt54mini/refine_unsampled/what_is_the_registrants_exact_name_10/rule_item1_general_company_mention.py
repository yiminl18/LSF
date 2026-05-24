def rule_item1_general_company_mention(doc: dict) -> list[dict]:
    """Match early Item 1 / General text spans that restate the company name."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = (span.get("structure", {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if "item 1" not in path and "business" not in path:
                continue
            if span.get("page_no", 999) > 12:
                continue
            if any(k in low for k in [
                "was incorporated",
                "is a holding company",
                "and its subsidiaries",
                "together with its subsidiaries",
                "hereinafter",
            ]):
                if any(ch.isalpha() for ch in txt):
                    out.append(span)
        return out
    except Exception:
        return []
