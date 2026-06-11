def rule_page1_cover_block_after_exact_name(doc: dict) -> list[dict]:
    """Match spans following '(Exact name of registrant...)' in the page-1 cover block."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r'Exact name of registrant', txt, re.I):
                for j in range(i, min(i + 10, len(texts))):
                    out.append(texts[j])
                break
        return out
    except Exception:
        return []
