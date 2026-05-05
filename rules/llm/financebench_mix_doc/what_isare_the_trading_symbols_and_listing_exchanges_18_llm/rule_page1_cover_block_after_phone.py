def rule_page1_cover_block_after_phone(doc: dict) -> list[dict]:
    """Match spans after the registrant phone line on page 1 where listing info usually starts."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and (
                "telephone number" in txt or "including area code" in txt
            ):
                for j in range(i + 1, min(len(texts), i + 12)):
                    s = texts[j]
                    if s.get("page_no") == 1:
                        out.append(s)
        return out
    except Exception:
        return []
