def rule_page1_h1_with_exact_name_in_next_five(doc: dict) -> list[dict]:
    """Match page-1 H1 spans with the exact-name caption appearing within the next five spans."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if not (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
            ):
                continue
            for j in range(i + 1, min(i + 6, len(texts))):
                if texts[j].get("page_no") != 1:
                    break
                if "exact name of registrant" in (texts[j].get("text", "") or "").lower():
                    out.append(span)
                    break
        return out
    except Exception:
        return []
