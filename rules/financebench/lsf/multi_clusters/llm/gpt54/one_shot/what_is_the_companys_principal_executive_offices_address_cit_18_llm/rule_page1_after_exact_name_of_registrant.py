def rule_page1_after_exact_name_of_registrant(doc: dict) -> list[dict]:
    """Match spans shortly after the '(Exact name of registrant...)' label on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if "exact name of registrant" in txt:
                for j in range(i + 1, min(len(texts), i + 8)):
                    cand = texts[j]
                    if cand.get("page_no") == 1:
                        out.append(cand)
        return out
    except Exception:
        return []
