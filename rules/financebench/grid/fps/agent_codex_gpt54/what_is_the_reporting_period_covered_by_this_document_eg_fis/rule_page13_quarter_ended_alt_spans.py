def rule_page13_quarter_ended_alt_spans(doc: dict) -> list[dict]:
    """Match alternate quarter-ended cover lines plus the first early quarter-results title/table block."""
    try:
        import re

        results = []
        month_date_re = (
            r"(january|february|march|april|may|june|july|august|september|"
            r"october|november|december)\s+\d{1,2},\s+\d{4}"
        )

        for span in doc.get("texts", []):
            if span.get("page_no", 999) > 3:
                continue

            lowered = " ".join(span.get("text", "").lower().split())
            if span.get("label") in ("text", "section_header") and lowered.startswith(
                "for the fiscal quarter ended"
            ) and re.search(month_date_re, lowered):
                results.append(span)
                continue

            if (
                span.get("page_no", 999) <= 2
                and span.get("label") in ("text", "section_header")
                and "quarter" in lowered
                and "financial results" in lowered
            ):
                results.append(span)
                continue

            if span.get("label") != "table":
                continue
            if "quarters ended" not in lowered:
                continue
            if not re.search(month_date_re, lowered):
                continue

            results.append(span)
            break

        return results
    except Exception:
        return []
