def rule_page1_main_header_children_after_section12b(doc: dict) -> list[dict]:
    """Match page-1 body children under the main company header after the Section 12(b) line."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and "section 12(b)" in (span.get("text") or "").lower():
                parent_path = ((span.get("structure") or {}).get("path_text") or "")
                for j in range(i + 1, min(len(texts), i + 8)):
                    s = texts[j]
                    if s.get("page_no") != 1:
                        continue
                    if ((s.get("structure") or {}).get("path_text") or "") == parent_path:
                        out.append(s)
        return out
    except Exception:
        return []
