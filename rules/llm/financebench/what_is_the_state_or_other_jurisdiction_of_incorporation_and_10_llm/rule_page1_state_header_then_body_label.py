def rule_page1_state_header_then_body_label(doc: dict) -> list[dict]:
    """Match page-1 state-value section headers whose child/body span is the incorporation label."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for idx, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                txt = (span.get("text") or "").strip()
                if txt and not re.search(r"\d", txt):
                    for child in texts:
                        if child.get("page_no") == 1 and child.get("structure", {}).get("parent_id") == idx:
                            ctext = (child.get("text") or "").strip()
                            if re.search(r"state( or other jurisdiction)? of incorporation( or organization)?|state of incorporation", ctext, re.I):
                                out.append(span)
                                out.append(child)
        return out
    except Exception:
        return []
