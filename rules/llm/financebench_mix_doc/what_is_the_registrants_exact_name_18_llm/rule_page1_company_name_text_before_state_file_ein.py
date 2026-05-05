def rule_page1_company_name_text_before_state_file_ein(doc: dict) -> list[dict]:
    """Match page-1 spans that precede a cluster of state, file number, and EIN identifiers."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            found = 0
            for j in range(i + 1, min(i + 20, len(texts))):
                s = texts[j]
                if s.get("page_no") != 1:
                    break
                t = ((s.get("text", "") or "") + " " + (s.get("text_span", "") or "")).lower()
                if "state or other jurisdiction" in t:
                    found += 1
                if "commission file" in t:
                    found += 1
                if "i.r.s. employer identification" in t or "irs employer identification" in t:
                    found += 1
            if found >= 2 and span.get("label") in {"text", "section_header"}:
                txt = (span.get("text", "") or "").lower()
                if "form 10-" not in txt and "form 8-k" not in txt and "securities and exchange commission" not in txt:
                    out.append(span)
        return out
    except Exception:
        return []
