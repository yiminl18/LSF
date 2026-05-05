def rule_page1_before_zip_code_label(doc: dict) -> list[dict]:
    """Match spans on page 1 immediately before '(Zip Code)' labels."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if "zip code" in txt:
                for j in range(max(0, i - 3), i):
                    cand = texts[j]
                    if cand.get("page_no") == 1:
                        out.append(cand)
        return out
    except Exception:
        return []
