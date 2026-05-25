def rule_cover_incorporation_ein(doc: dict) -> list[dict]:
    '''Page-1 cover-page spans matching the State-of-incorporation / I.R.S. Employer Identification label phrases, the spans immediately preceding them (which carry the actual state name and EIN value), and any EIN-format (NN-NNNNNNN) span on page 1.'''
    import re
    texts = doc.get("texts", [])
    label_re = re.compile(
        r"(state\s+(?:or\s+other\s+jurisdiction\s+of\s+incorporation|of\s+incorporation)"
        r"|i\.?\s*r\.?\s*s\.?\s+employer\s+identification"
        r"|irs\s+employer\s+identification)",
        re.IGNORECASE,
    )
    ein_re = re.compile(r"\b\d{2}-\d{7}\b")
    selected = set()
    for i, span in enumerate(texts):
        if span.get("page_no") != 1:
            continue
        text = span.get("text", "") or ""
        if label_re.search(text):
            selected.add(i)
            j = i - 1
            while j >= 0 and texts[j].get("page_no") == 1:
                prev_text = texts[j].get("text", "") or ""
                if prev_text.strip() and not label_re.search(prev_text):
                    selected.add(j)
                    break
                j -= 1
        if ein_re.search(text):
            selected.add(i)
    return [texts[i] for i in sorted(selected)]
