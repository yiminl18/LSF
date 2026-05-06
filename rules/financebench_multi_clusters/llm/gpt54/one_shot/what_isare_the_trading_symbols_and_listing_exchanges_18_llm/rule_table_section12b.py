def rule_table_section12b(doc: dict) -> list[dict]:
    """Match page-1 tables under the Section 12(b) registration area."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("label") == "table" and span.get("page_no") == 1:
                path = ((span.get("structure") or {}).get("path_text") or "").lower()
                txt = (span.get("text") or "").lower()
                if (
                    "section 12(b)" in path
                    or "section 12(b)" in txt
                    or "trading symbol" in txt
                    or "name of each exchange on which registered" in txt
                ):
                    out.append(span)
        return out
    except Exception:
        return []
