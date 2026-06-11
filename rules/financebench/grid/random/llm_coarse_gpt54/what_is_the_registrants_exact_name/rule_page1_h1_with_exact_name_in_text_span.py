def rule_page1_h1_with_exact_name_in_text_span(doc: dict) -> list[dict]:
    """Match page-1 H1 section headers whose text_span contains the exact-name caption."""
    try:
        out = []
        for span in doc.get("texts", []):
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
            ):
                tspan = (span.get("text_span") or "").lower()
                if "exact name of registrant as specified in its charter" in tspan:
                    out.append(span)
        return out
    except Exception:
        return []
