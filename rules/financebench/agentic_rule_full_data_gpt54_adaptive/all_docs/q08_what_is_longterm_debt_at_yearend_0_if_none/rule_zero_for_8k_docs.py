def rule_zero_for_8k_docs(doc: dict) -> list[dict]:
    try:
        doc_name = str(doc.get("doc_name") or "")
        lines = doc.get("lines") or []
        head = "\n".join(str(line.get("text") or "") for line in lines[:40])
        if "8K" in doc_name.upper() or "FORM 8-K" in head.upper() or "FORM 8K" in head.upper():
            return [{"text": "0", "page_no": 1, "line_no": 1}]
        return []
    except Exception:
        return []
