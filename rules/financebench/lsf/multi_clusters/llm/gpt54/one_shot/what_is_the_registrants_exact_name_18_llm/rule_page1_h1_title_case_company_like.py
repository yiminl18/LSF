def rule_page1_h1_title_case_company_like(doc: dict) -> list[dict]:
    """Match page-1 H1 section headers in title case that precede the exact-name caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            txt = (span.get("text", "") or "").strip()
            low = txt.lower()
            if not (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "form 10-" not in low
                and "form 8-k" not in low
                and "current report" not in low
                and "securities and exchange commission" not in low
            ):
                continue
            nxt = texts[i + 1]
            if "exact name of registrant" in (nxt.get("text", "") or "").lower():
                out.append(span)
        return out
    except Exception:
        return []
