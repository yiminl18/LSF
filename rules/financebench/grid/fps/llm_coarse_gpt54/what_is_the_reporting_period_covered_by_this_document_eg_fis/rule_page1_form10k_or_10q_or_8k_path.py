def rule_page1_form10k_or_10q_or_8k_path(doc: dict) -> list[dict]:
    """Match any page-1 span whose path_text explicitly references FORM 10-K, 10-Q, or 8-K and contains date language."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            text = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and any(f in path for f in ["form 10-k", "form 10-q", "form 8-k"]):
                if any(k in text for k in ["ended", "date of report", "event reported", "period ending"]):
                    out.append(span)
        return out
    except Exception:
        return []
