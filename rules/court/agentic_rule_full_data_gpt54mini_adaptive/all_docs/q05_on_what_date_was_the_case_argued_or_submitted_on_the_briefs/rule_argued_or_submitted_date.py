import re


def rule_argued_or_submitted_date(doc: dict) -> list[dict]:
    try:
        def _intish(value, default):
            try:
                if value is None:
                    return default
                return int(value)
            except Exception:
                return default

        month_pat = (
            r"(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
        )
        date_pat = month_pat + r"\s+\d{1,2},\s+\d{4}"
        header_pat = re.compile(
            r"^\s*((?:Argued(?:\s+and)?\s+Submitted|Argued|Submitted)"
            r"(?:\s+En Banc)?\s+" + date_pat + r")\b",
            re.IGNORECASE,
        )
        oral_pat = re.compile(
            r"(\bheard\s+oral\s+argument\b.*?\bon\s+" + date_pat + r")\b",
            re.IGNORECASE | re.DOTALL,
        )
        header_any_pat = re.compile(
            r"(?:^|\n)\s*((?:Argued(?:\s+and)?\s+Submitted|Argued|Submitted)"
            r"(?:\s+En Banc)?\s+" + date_pat + r")\b",
            re.IGNORECASE,
        )
        oral_any_pat = re.compile(
            r"(\bheard\s+oral\s+argument\b.*?\bon\s+" + date_pat + r")\b",
            re.IGNORECASE | re.DOTALL,
        )

        lines = doc.get("lines") or []
        ordered = sorted(
            lines,
            key=lambda x: (
                _intish(x.get("page_no"), 10**9),
                _intish(x.get("line_no"), 10**9),
            ),
        )

        for item in ordered[:120]:
            text = (item.get("text") or "").replace("\x0c", " ").strip()
            if not text:
                continue

            m = header_pat.search(text)
            if m:
                return [
                    {
                        "text": m.group(1).strip(),
                        "page_no": item.get("page_no"),
                        "line_no": item.get("line_no"),
                    }
                ]

            if "oral argument" in text.lower():
                m = oral_pat.search(text)
                if m:
                    return [
                        {
                            "text": m.group(1).strip(),
                            "page_no": item.get("page_no"),
                            "line_no": item.get("line_no"),
                        }
                    ]

        paragraphs = doc.get("paragraphs") or []
        ordered_paragraphs = sorted(
            paragraphs,
            key=lambda x: (
                _intish(x.get("page_no"), 10**9),
                _intish(x.get("paragraph_no"), 10**9),
            ),
        )

        for item in ordered_paragraphs[:40]:
            text = (item.get("text") or "").replace("\x0c", " ").strip()
            if not text:
                continue

            m = header_pat.search(text)
            if m:
                result = {"text": m.group(1).strip()}
                if item.get("page_no") is not None:
                    result["page_no"] = item.get("page_no")
                if item.get("paragraph_no") is not None:
                    result["paragraph_no"] = item.get("paragraph_no")
                return [result]

            m = oral_pat.search(text)
            if m:
                result = {"text": m.group(1).strip()}
                if item.get("page_no") is not None:
                    result["page_no"] = item.get("page_no")
                if item.get("paragraph_no") is not None:
                    result["paragraph_no"] = item.get("paragraph_no")
                return [result]

        full_text = (doc.get("text") or "").replace("\x0c", " ")
        for blob in (
            full_text[:8000],
            "\n".join((p.get("text") or "").replace("\x0c", " ") for p in (doc.get("pages") or [])[:2]),
        ):
            if not blob:
                continue

            m = header_any_pat.search(blob)
            if m:
                return [{"text": m.group(1).strip()}]

            m = oral_any_pat.search(blob)
            if m:
                return [{"text": m.group(1).strip()}]

        return []
    except Exception:
        return []
