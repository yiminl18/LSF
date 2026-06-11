def rule_page1_company_heading_with_following_caption_body(doc: dict) -> list[dict]:
    """Match page-1 headings whose immediate child/body contains the exact-name caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            if span.get("bold") != 1:
                continue
            if float(span.get("size") or 0) < 10:
                continue
            sid = i
            for child in texts[i+1:i+4]:
                if child.get("structure", {}).get("parent_id") == sid and "exact name of registrant" in ((child.get("text") or "").lower()):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
