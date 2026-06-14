def rule_notice_conclusion_parenthetical_citations(doc: dict) -> list[dict]:
    """Match the shortest conclusion-style notice line per path when it names an exact parenthetical 49 CFR subsection."""
    try:
        import re

        texts = doc.get("texts", [])
        stop_re = re.compile(r"^\s*response to this notice\s*$", re.IGNORECASE)
        order_item_re = re.compile(r"\b(in regard to item|with respect to item|regarding item)\b", re.IGNORECASE)
        exact_sub_re = re.compile(
            r"(?:§|s)\s*(19\d\.\d+[a-z0-9.-]*(?:\([^)]+\))+)",
            re.IGNORECASE,
        )
        conclusion_re = re.compile(
            r"\btherefore\b|not in compliance|accordingly,?\s+.*violation of",
            re.IGNORECASE,
        )

        cutoff = len(texts)
        for i, span in enumerate(texts):
            text = " ".join((span.get("text") or "").split())
            if stop_re.match(text):
                cutoff = i
                break
            if span.get("label") in {"text", "list_item"} and order_item_re.search(text):
                cutoff = i
                break

        best_by_path = {}
        for span in texts[:cutoff]:
            if span.get("label") not in {"text", "list_item"}:
                continue
            text = span.get("text") or ""
            if order_item_re.search(text):
                continue
            if not exact_sub_re.search(text):
                continue
            if not conclusion_re.search(text):
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").strip()
            key = path or f"page:{span.get('page_no', 0)}"
            score = (len(text), len(text.split()))
            if key not in best_by_path or score < best_by_path[key][0]:
                best_by_path[key] = (score, span)
        return [item[1] for item in best_by_path.values()]
    except Exception:
        return []
