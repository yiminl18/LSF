def rule_page1_near_exact_name_caption_window(doc: dict) -> list[dict]:
    """Match spans within a short backward window from the exact-name caption on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            if "exact name of registrant" in (span.get("text", "") or "").lower():
                for j in range(max(0, i - 3), i):
                    cand = texts[j]
                    if cand.get("page_no") == 1 and cand.get("label") in {"text", "section_header"}:
                        out.append(cand)
        return out
    except Exception:
        return []
