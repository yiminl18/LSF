def rule_page1_h1_before_address_block(doc: dict) -> list[dict]:
    """Match page-1 H1 company headers followed by principal executive office address block."""
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
            found_addr = False
            for j in range(i + 1, min(i + 15, len(texts))):
                s = texts[j]
                if s.get("page_no") != 1:
                    break
                t = (s.get("text", "") or "").lower()
                if "address of principal executive offices" in t or "address and telephone number" in t:
                    found_addr = True
                    break
            if found_addr:
                out.append(span)
        return out
    except Exception:
        return []
