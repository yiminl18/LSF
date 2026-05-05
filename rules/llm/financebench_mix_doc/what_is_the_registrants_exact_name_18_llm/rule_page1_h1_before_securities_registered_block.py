def rule_page1_h1_before_securities_registered_block(doc: dict) -> list[dict]:
    """Match page-1 H1 company headers followed by 'Securities registered pursuant to Section 12(b)'."""
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
            found_sec = False
            for j in range(i + 1, min(i + 25, len(texts))):
                s = texts[j]
                if s.get("page_no") != 1:
                    break
                t = ((s.get("text", "") or "") + " " + (s.get("text_span", "") or "")).lower()
                if "securities registered pursuant to section 12(b)" in t:
                    found_sec = True
                    break
            if found_sec:
                out.append(span)
        return out
    except Exception:
        return []
