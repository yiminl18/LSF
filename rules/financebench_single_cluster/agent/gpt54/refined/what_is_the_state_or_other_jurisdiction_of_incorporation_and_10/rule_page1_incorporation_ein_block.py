def rule_page1_incorporation_ein_block(doc: dict) -> list[dict]:
    """Retrieve page-1 registrant incorporation/EIN spans and adjacent company header spans."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            label = span.get("label") or ""
            page_no = span.get("page_no")
            if page_no != 1:
                continue
            t = text.lower()
            p = path.lower()
            hit = False
            if "state or other jurisdiction of incorporation" in t or "state of incorporation" in t:
                hit = True
            if "i.r.s. employer identification" in t or "irs employer identification" in t or "employer identification no" in t:
                hit = True
            if "state or other jurisdiction of incorporation" in p or "state of incorporation" in p:
                hit = True
            if "i.r.s. employer identification" in p or "irs employer identification" in p or "employer identification no" in p:
                hit = True
            if re.fullmatch(r"\d{2}-\d{7}", text.strip()):
                hit = True
            if hit:
                for j in range(max(0, i-2), min(len(texts), i+3)):
                    s2 = texts[j]
                    if s2.get("page_no") == 1 and s2 not in out:
                        out.append(s2)
        return out
    except Exception:
        return []

