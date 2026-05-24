def rule_page1_company_name_in_path_text_of_following_spans(doc: dict) -> list[dict]:
    """Match page-1 H1 spans whose text is reused as path_text prefix for many following page-1 spans."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if not (span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H1" and txt):
                continue
            count = 0
            for j in range(i + 1, min(i + 30, len(texts))):
                nxt = texts[j]
                if nxt.get("page_no") != 1:
                    break
                p = nxt.get("structure", {}).get("path_text") or ""
                if p.startswith(txt):
                    count += 1
            if count >= 3:
                out.append(span)
        return out
    except Exception:
        return []
