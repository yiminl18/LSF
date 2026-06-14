def rule_pco_final_order_deadline_list_items(doc: dict) -> list[dict]:
    """Match PCO list items with Final Order deadlines, else fall back to top-level PCO action items."""
    try:
        import re

        texts = doc.get("texts", [])

        def in_pco_path(span: dict) -> bool:
            path = (((span.get("structure") or {}).get("path_text")) or "").upper()
            return "PROPOSED COMPLIANCE ORDER" in path

        def near_pco_heading(idx: int, window: int = 8) -> bool:
            page = texts[idx].get("page_no")
            for j in range(max(0, idx - window), idx):
                prev_text = (texts[j].get("text") or "").strip().upper()
                if prev_text == "PROPOSED COMPLIANCE ORDER" and texts[j].get("page_no") in {page, page - 1}:
                    return True
            return False

        deadline_items = []
        for i, span in enumerate(texts):
            text = (span.get("text") or "").lower()
            if (
                span.get("label") == "list_item"
                and "final order" in text
                and (in_pco_path(span) or near_pco_heading(i))
            ):
                deadline_items.append(span)
        if deadline_items:
            return deadline_items

        top_alpha_re = re.compile(r"^\s*[A-Z](?:\.|\))\s+")
        top_num_re = re.compile(r"^\s*\d+\.\s+")

        fallback_items = []
        for i, span in enumerate(texts):
            text = (span.get("text") or "").strip()
            text_lower = text.lower()
            if span.get("label") != "list_item":
                continue
            if not (in_pco_path(span) or near_pco_heading(i)):
                continue
            if "requested" in text_lower and "not mandated" in text_lower:
                continue
            if top_alpha_re.match(text) or top_num_re.match(text):
                fallback_items.append(span)
        return fallback_items
    except Exception:
        return []
