def rule_budget_year_percent(doc: dict) -> list[dict]:
    try:
        import re

        spans: list[dict] = []
        seen = set()
        paragraphs = doc.get("paragraphs") or []

        def compact(s: str) -> str:
            return re.sub(r"[^a-z0-9.]+", "", s.lower())

        percent_re = re.compile(
            r"(?i)\b\d+(?:\.\d+)?\s*(?:%|percent|per cent)\s*(?:of|as a share of|share of|in relation to|relative to)?\s*(?:gdp|gnp|nominal gdp|gross national product)\b"
        )
        year_re = re.compile(r"(?i)\b(?:fy|fiscal(?: year)?)\s*\d{2,4}\b")

        for para in paragraphs:
            text = (para.get("text") or "").strip()
            low = text.lower()
            ctext = compact(text)
            if not text:
                continue
            if not percent_re.search(text) and not re.search(
                r"\b\d+(?:\.\d+)?(?:%|percent|percentof|percentinrelationto|per cent)\b.*\b(?:gdp|gnp|nominalgdp|grossnationalproduct)\b",
                ctext,
            ):
                continue
            if not year_re.search(text) and "fiscal" not in low and "fy" not in low:
                continue
            if not any(
                term in low or term.replace(" ", "") in ctext
                for term in (
                    "budget",
                    "deficit",
                    "surplus",
                    "outlays",
                    "receipts",
                    "federal budget",
                    "federal deficit",
                    "federal outlays",
                    "federal receipts",
                )
            ):
                continue
            key = (
                para.get("page_no"),
                para.get("paragraph_no"),
                text,
            )
            if key in seen:
                continue
            seen.add(key)
            spans.append({k: v for k, v in para.items() if k in {"page_no", "paragraph_no", "text"}})

        return spans
    except Exception:
        return []
