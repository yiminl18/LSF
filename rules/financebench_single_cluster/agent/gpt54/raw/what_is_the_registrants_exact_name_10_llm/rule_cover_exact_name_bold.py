def rule_cover_exact_name_bold(doc: dict) -> list[dict]:
    """Retrieve bold cover-page company-name spans near the exact-name-of-registrant cue."""
    try:
        import re
        texts = doc.get("texts", []) or []
        out = []
        n = len(texts)
        for i, span in enumerate(texts):
            if not isinstance(span, dict):
                continue
            if span.get("page_no") != 1:
                continue
            if span.get("bold") != 1:
                continue
            if span.get("label") not in {"section_header", "text"}:
                continue
            txt = (span.get("text") or "").strip()
            if not txt or len(txt) > 120:
                continue
            if re.search(r"exact name of registrant", txt, re.I):
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            window = texts[max(0, i-3):min(n, i+4)]
            nearby = " ".join((s.get("text") or "") for s in window if isinstance(s, dict))
            if re.search(r"exact name of registrant", nearby, re.I) or re.search(r"exact name of registrant", path, re.I):
                out.append(span)
                continue
            if span.get("label") == "section_header" and ((span.get("structure") or {}).get("level") in {"H1", "H2"}):
                if any(re.search(r"exact name of registrant", (texts[j].get("text") or ""), re.I) for j in range(i+1, min(n, i+4)) if isinstance(texts[j], dict)):
                    out.append(span)
        return out
    except Exception:
        return []

