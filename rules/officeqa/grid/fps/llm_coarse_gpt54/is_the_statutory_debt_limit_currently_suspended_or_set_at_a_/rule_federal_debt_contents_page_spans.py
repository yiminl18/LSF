def rule_federal_debt_contents_page_spans(doc: dict) -> list[dict]:
    """Return all contents-page spans under Federal Debt, useful because the target table/page is consistently listed there."""
    try:
        out = []
        in_fd = False
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "contents" not in path.lower() and text.strip().lower() != "federal debt":
                continue
            if span.get("label") == "section_header" and text.strip().lower() == "federal debt":
                in_fd = True
                out.append(span)
                continue
            if in_fd:
                if span.get("label") == "section_header" and text.strip() and text.strip().lower() != "federal debt":
                    if span.get("page_no") == out[0].get("page_no"):
                        break
                if span.get("page_no") == out[0].get("page_no"):
                    out.append(span)
        return out
    except Exception:
        return []
