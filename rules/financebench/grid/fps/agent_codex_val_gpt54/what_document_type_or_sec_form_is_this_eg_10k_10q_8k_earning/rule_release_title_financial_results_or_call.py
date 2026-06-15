def rule_release_title_financial_results_or_call(doc: dict) -> list[dict]:
    """Match news-release titles or financial-results headings that identify earnings-style releases."""
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("label") not in {"text", "section_header"}:
                continue

            text = " ".join((s.get("text") or "").split())
            lowered = text.lower()
            page = s.get("page_no") or 99

            if page <= 2 and lowered == "news release":
                out.append(s)
                continue

            if "earnings results conference call" in lowered:
                out.append(s)
                continue

            if s.get("bold") == 1 and "financial results" in lowered and page <= 6:
                out.append(s)

        return out
    except Exception:
        return []
