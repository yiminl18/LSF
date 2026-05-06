def rule_page1_h1_before_telephone_block(doc: dict) -> list[dict]:
    """Match page-1 H1 company headers followed by registrant telephone caption."""
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
            found_phone = False
            for j in range(i + 1, min(i + 20, len(texts))):
                s = texts[j]
                if s.get("page_no") != 1:
                    break
                t = (s.get("text", "") or "").lower()
                if "registrant" in t and "telephone number" in t:
                    found_phone = True
                    break
            if found_phone:
                out.append(span)
        return out
    except Exception:
        return []
