def rule_notice_shorthand_parenthetical_expansions(doc: dict) -> list[dict]:
    """Match allegation lines that spell out a subsection omitted by a shorthand conclusion like '§ 195.402(a) & (c)(13)'."""
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
        allegation_re = re.compile(
            r"\bfailed\b|in accordance with|as required by|required by|pursuant to|comply with|in violation of",
            re.IGNORECASE,
        )
        shorthand_re = re.compile(r"(?:&|and)\s*\(", re.IGNORECASE)

        cutoff = len(texts)
        for i, span in enumerate(texts):
            text = " ".join((span.get("text") or "").split())
            if stop_re.match(text):
                cutoff = i
                break
            if span.get("label") in {"text", "list_item"} and order_item_re.search(text):
                cutoff = i
                break

        conclusion_text_by_path = {}
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
            if key not in conclusion_text_by_path or score < conclusion_text_by_path[key][0]:
                conclusion_text_by_path[key] = (score, text.lower())

        best_by_path = {}
        for span in texts[:cutoff]:
            if span.get("label") not in {"text", "list_item"}:
                continue
            text = span.get("text") or ""
            if order_item_re.search(text):
                continue
            if not exact_sub_re.search(text):
                continue
            if conclusion_re.search(text):
                continue
            if not allegation_re.search(text):
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").strip()
            key = path or f"page:{span.get('page_no', 0)}"
            conclusion_item = conclusion_text_by_path.get(key)
            if not conclusion_item:
                continue
            conclusion_text = conclusion_item[1]
            if not shorthand_re.search(conclusion_text):
                continue
            citations = [m.group(1).lower() for m in exact_sub_re.finditer(text)]
            if citations and all(citation in conclusion_text for citation in citations):
                continue
            score = (len(text), len(text.split()))
            if key not in best_by_path or score < best_by_path[key][0]:
                best_by_path[key] = (score, span)
        return [item[1] for item in best_by_path.values()]
    except Exception:
        return []
