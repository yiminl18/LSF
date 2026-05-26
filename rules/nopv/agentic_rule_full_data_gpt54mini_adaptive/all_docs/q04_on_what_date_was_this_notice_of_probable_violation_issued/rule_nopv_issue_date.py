import re


def rule_nopv_issue_date(doc: dict) -> list[dict]:
    try:
        month_day_year_pat = re.compile(
            r"(?i)\b(?:january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\s+\d{1,2}\s*,\s*\d{4}\b"
        )
        lines = doc.get("lines") or []
        ordered = sorted(
            lines,
            key=lambda x: (
                x.get("page_no", 10**9),
                x.get("line_no", 10**9),
            ),
        )

        header_anchor_idx = None
        for idx, item in enumerate(ordered[:60]):
            text = (item.get("text") or "").strip()
            low = text.lower()
            if not low:
                continue
            if "notice of probable violation" in low:
                header_anchor_idx = idx
                break
            if "notice of probable" in low:
                header_anchor_idx = idx
                break

        search_end = len(ordered)
        for idx, item in enumerate(ordered[:80]):
            if (item.get("text") or "").strip().lower().startswith("dear "):
                search_end = idx
                break

        if header_anchor_idx is not None:
            for item in ordered[header_anchor_idx:search_end]:
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                if month_day_year_pat.fullmatch(text) or month_day_year_pat.search(text):
                    return [
                        {
                            "text": text,
                            "page_no": item.get("page_no"),
                            "line_no": item.get("line_no"),
                        }
                    ]

        return [{"text": "NOT FOUND"}]
    except Exception:
        return []
