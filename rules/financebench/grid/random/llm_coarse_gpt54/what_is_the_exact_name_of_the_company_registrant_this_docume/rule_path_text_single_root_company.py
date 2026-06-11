def rule_path_text_single_root_company(doc: dict) -> list[dict]:
    """Match root H1 spans whose path_text equals their text and whose descendants contain registrant cover-page fields."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "").strip()
            path = span.get("structure", {}).get("path_text") or ""
            if not (
                span.get("page_no") == 1
                and span.get("structure", {}).get("level") == "H1"
                and txt
                and path == txt
            ):
                continue
            hits = 0
            for s in texts:
                spath = s.get("structure", {}).get("path_text") or ""
                stxt = (s.get("text") or "").lower()
                if spath == path or spath.startswith(path + " |"):
                    if any(k in stxt for k in [
                        "exact name of registrant",
                        "state or other jurisdiction",
                        "employer identification",
                        "address of principal executive offices",
                        "registrant’s telephone number",
                        "registrant's telephone number",
                    ]):
                        hits += 1
            if hits >= 2:
                out.append(span)
        return out
    except Exception:
        return []
