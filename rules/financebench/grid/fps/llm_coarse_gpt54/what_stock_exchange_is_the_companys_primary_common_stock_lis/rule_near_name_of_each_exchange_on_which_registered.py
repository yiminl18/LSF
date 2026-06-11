def rule_near_name_of_each_exchange_on_which_registered(doc: dict) -> list[dict]:
    """Match spans on page 1 near the phrase 'Name of each exchange on which registered'."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        anchor_idxs = []
        for i, span in enumerate(texts):
            combined = " ".join([
                span.get("text", "") or "",
                span.get("text_span", "") or "",
                span.get("structure", {}).get("path_text", "") or "",
            ])
            if re.search(r'name of each exchange on which registered', combined, re.I):
                anchor_idxs.append(i)
        for idx in anchor_idxs:
            for j in range(max(0, idx - 3), min(len(texts), idx + 6)):
                s = texts[j]
                if s.get("page_no") == texts[idx].get("page_no", 1):
                    t = (s.get("text") or "").strip()
                    if t and not re.search(r'name of each exchange on which registered', t, re.I):
                        out.append(s)
        return out
    except Exception:
        return []
