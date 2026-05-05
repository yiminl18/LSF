def rule_h1_page1_with_exact_name_child(doc: dict) -> list[dict]:
    """Match page-1 H1 spans that have descendants/body siblings under them containing the exact-name parenthetical."""
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
            path = span.get("structure", {}).get("path_text") or ""
            txt = span.get("text") or ""
            if "FORM 10-" in txt or "FORM 8-K" in txt or "CURRENT REPORT" in txt:
                continue
            found = False
            for j in range(i + 1, min(i + 12, len(texts))):
                nxt = texts[j]
                if nxt.get("page_no") != 1:
                    break
                if "(Exact name of registrant as specified in its charter)" in (nxt.get("text") or ""):
                    found = True
                    break
            if found or "(Exact name of registrant as specified in its charter)" in txt:
                out.append(span)
        return out
    except Exception:
        return []
