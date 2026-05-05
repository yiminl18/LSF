def rule_page1_text_near_exact_name_of_registrant(doc: dict) -> list[dict]:
    """Match page-1 spans under the company cover block path_text, near the exact-name-of-registrant area."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "") or ""
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and (
                "exact name of registrant" in text.lower() or path.strip() != "" and span.get("structure", {}).get("depth", 0) in (2, 3)
            ):
                if "commission" not in path.lower() and "form 10-k" not in path.lower():
                    out.append(span)
        return out
    except Exception:
        return []
