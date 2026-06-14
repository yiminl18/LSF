def rule_exact_pco_top_level_items(doc: dict) -> list[dict]:
    """Match top-level list items in exact PCO sections, with narrow requested-line suppression."""
    try:
        import re

        texts = doc.get("texts", [])
        items = [
            s for s in texts
            if s.get("label") == "list_item"
            and (((s.get("structure") or {}).get("path_text") or "").strip().upper() == "PROPOSED COMPLIANCE ORDER")
        ]
        if not items:
            return []

        top_alpha_re = re.compile(r"^\s*[A-Z](?:\.|)\s+(?=[A-Z(])")
        top_num_re = re.compile(r"^\s*\d+\.\s+")
        lower_alpha_re = re.compile(r"^\s*[a-z](?:\.|)\s+(?=[A-Za-z(])")
        roman_re = re.compile(r"^\s*(?:\([ivxlcdm]+\)|[ivxlcdm]{2,}\.)", re.I)
        plain_noise_re = re.compile(r"^\s*[`•]")
        requested_re = re.compile(r"\bit is requested\b|\bnot mandated\b", re.I)
        intro_re = re.compile(r":\s*$|must do the following:|must:$", re.I)
        ref_item_letter_re = re.compile(r"\bitem\s+[A-Z]\b|\bitem\s+[A-Z]\s+above\b", re.I)
        notice_item_re = re.compile(r"\bitem(?: number)?\s+\d+\b|\bnumber\s+\d+\b", re.I)

        has_alpha = any(top_alpha_re.match((s.get("text") or "").strip()) for s in items)
        has_num = any(top_num_re.match((s.get("text") or "").strip()) for s in items)
        if has_alpha:
            top = [s for s in items if top_alpha_re.match((s.get("text") or "").strip())]
        elif has_num:
            top = [s for s in items if top_num_re.match((s.get("text") or "").strip())]
        else:
            top = [
                s for s in items
                if not lower_alpha_re.match((s.get("text") or "").strip())
                and not roman_re.match((s.get("text") or "").strip())
            ]

        intros = [
            s for s in top
            if not requested_re.search((s.get("text") or "").strip())
            and intro_re.search((s.get("text") or "").strip())
        ]
        if len(intros) == 1:
            intro_span = intros[0]
            start = items.index(intro_span) + 1
            collected = []
            skip_nested = False
            saw_lower = False
            for span in items[start:]:
                text = (span.get("text") or "").strip()
                if span in top:
                    break
                if not text or plain_noise_re.match(text):
                    continue
                if "may consider" in text.lower():
                    continue
                if skip_nested and not lower_alpha_re.match(text):
                    continue
                if roman_re.match(text) and saw_lower:
                    continue
                collected.append(span)
                if lower_alpha_re.match(text):
                    saw_lower = True
                    skip_nested = text.endswith(":")
            others = []
            for span in top:
                if span is intro_span:
                    continue
                text = (span.get("text") or "").strip()
                if requested_re.search(text):
                    continue
                if ref_item_letter_re.search(text) and not notice_item_re.search(text):
                    continue
                others.append(span)
            if len(collected) >= 2:
                return []

        intro_count = sum(
            1 for s in top
            if not requested_re.search((s.get("text") or "").strip())
            and intro_re.search((s.get("text") or "").strip())
        )
        nonintro_count = sum(
            1 for s in top
            if not requested_re.search((s.get("text") or "").strip())
            and not intro_re.search((s.get("text") or "").strip())
        )
        if nonintro_count == 0 or (intro_count >= 3 and nonintro_count <= 2):
            return [s for s in top if not requested_re.search((s.get("text") or "").strip())]
        return top
    except Exception:
        return []
