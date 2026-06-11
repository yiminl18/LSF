def rule_page1_h1_with_company_suffix_or_comma(doc: dict) -> list[dict]:
    """Match page-1 H1 spans containing common company suffixes or comma-inc style names."""
    try:
        import re
        out = []
        pat = re.compile(r"(,?\sinc\.?$|\bcorporation\b|\bcompany\b|\bplc\b|\bltd\.?\b)", re.I)
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("structure", {}).get("level") == "H1"
                and pat.search(txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
