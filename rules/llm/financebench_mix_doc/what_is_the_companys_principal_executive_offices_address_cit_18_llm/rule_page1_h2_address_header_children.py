def rule_page1_h2_address_header_children(doc: dict) -> list[dict]:
    """Match child spans under page 1 H2 address headers, useful when city/state is split across spans."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        address_paths = set()
        for span in texts:
            if span.get("page_no") == 1 and span.get("label") == "section_header" and span.get("structure", {}).get("level") == "H2":
                txt = span.get("text") or ""
                if re.search(r"^\d{1,5}\s", txt) or "warmley" in txt.lower() or "bristol" in txt.lower():
                    address_paths.add(span.get("structure", {}).get("path_text"))
        for span in texts:
            if span.get("page_no") != 1:
                continue
            if span.get("structure", {}).get("path_text") in address_paths and span.get("structure", {}).get("level") == "Body":
                out.append(span)
        return out
    except Exception:
        return []
