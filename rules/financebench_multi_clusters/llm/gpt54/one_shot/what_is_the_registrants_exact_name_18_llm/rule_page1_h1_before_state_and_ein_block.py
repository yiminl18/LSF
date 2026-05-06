def rule_page1_h1_before_state_and_ein_block(doc: dict) -> list[dict]:
    """Match page-1 H1 company headers that are followed by state and EIN information block."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if not (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
            ):
                continue
            found_state = False
            found_ein = False
            for j in range(i + 1, min(i + 12, len(texts))):
                s = texts[j]
                if s.get("page_no") != 1:
                    break
                t = (s.get("text", "") or "").lower()
                if "state or other jurisdiction of incorporation" in t or "state or other jurisdiction of incorporation or organization" in t:
                    found_state = True
                if "i.r.s. employer identification" in t or "irs employer identification" in t:
                    found_ein = True
            if found_state and found_ein:
                out.append(span)
        return out
    except Exception:
        return []
