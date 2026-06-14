def rule_page1_case_number_after_dear(doc: dict) -> list[dict]:
    """Match the page-1 case-number line that follows the Dear salutation."""
    try:
        import re

        texts = doc.get("texts", [])
        case_re = re.compile(r"\b(?:CPF\s*)?\d-\d{4}-\d{3}\s*-?\s*NOPV\b", re.IGNORECASE)
        hits = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            if not case_re.search((span.get("text") or "")):
                continue
            for j in range(max(0, i - 3), i):
                prev_text = (texts[j].get("text") or "").strip().lower()
                if texts[j].get("page_no") == 1 and prev_text.startswith("dear "):
                    hits.append(span)
                    break
        return hits
    except Exception:
        return []
