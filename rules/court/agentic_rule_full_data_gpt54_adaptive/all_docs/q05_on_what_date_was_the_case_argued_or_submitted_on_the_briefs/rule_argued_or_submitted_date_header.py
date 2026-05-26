import re


def rule_argued_or_submitted_date_header(doc: dict) -> list[dict]:
    try:
        month_pat = (
            r"(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
        )
        date_pat = rf"{month_pat}\s+\d{{1,2}},\s+\d{{4}}"
        header_patterns = (
            re.compile(
                rf"^\s*(Argued\s+and\s+Submitted(?:\s+En\s+Banc)?\s+{date_pat}(?:\s*\*+)?)\s*$",
                re.IGNORECASE,
            ),
            re.compile(
                rf"^\s*(Submitted(?:\s+on\s+the\s+briefs)?(?:\s+En\s+Banc)?\s+{date_pat}(?:\s*\*+)?)\s*$",
                re.IGNORECASE,
            ),
            re.compile(
                rf"^\s*(Argued(?:\s+En\s+Banc)?\s+{date_pat}(?:\s*\*+)?)\s*$",
                re.IGNORECASE,
            ),
        )
        negative_prefixes = (
            "submission deferred",
            "submission vacated",
            "submission withdrawn",
            "resubmitted",
        )

        def _intish(value, default):
            try:
                return int(value)
            except Exception:
                return default

        def _clean(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").replace("\x0c", " ")).strip()

        def _extract(text: str) -> str | None:
            cleaned = _clean(text)
            if not cleaned:
                return None
            lowered = cleaned.lower()
            if lowered.startswith(negative_prefixes):
                return None
            for pattern in header_patterns:
                match = pattern.match(cleaned)
                if match:
                    return match.group(1).strip()
            return None

        def _scan_items(items: list[dict], line_field: str, limit: int) -> list[dict]:
            ordered = sorted(
                [item for item in (items or []) if isinstance(item, dict)],
                key=lambda item: (
                    _intish(item.get("page_no"), 10**9),
                    _intish(item.get(line_field), 10**9),
                ),
            )
            scan_limit = min(len(ordered), limit)
            for i in range(scan_limit):
                item = ordered[i]
                text = _clean(item.get("text", ""))
                if not text:
                    continue

                for end in range(i + 1, min(i + 4, scan_limit + 1)):
                    combined = " ".join(
                        _clean(ordered[j].get("text", ""))
                        for j in range(i, end)
                        if _clean(ordered[j].get("text", ""))
                    )
                    match_text = _extract(combined)
                    if not match_text:
                        continue
                    span = {"text": match_text}
                    if item.get("page_no") is not None:
                        span["page_no"] = item.get("page_no")
                    if item.get(line_field) is not None:
                        span[line_field] = item.get(line_field)
                    return [span]
            return []

        line_hits = _scan_items(doc.get("lines") or [], "line_no", 320)
        if line_hits:
            return line_hits

        paragraph_hits = _scan_items(doc.get("paragraphs") or [], "paragraph_no", 160)
        if paragraph_hits:
            return paragraph_hits

        full_text = (doc.get("text") or "").replace("\x0c", " ")
        top_blob = full_text[:25000]
        matches = []
        for pattern in header_patterns:
            for match in pattern.finditer(top_blob):
                matches.append((match.start(), match.group(1).strip()))
        if matches:
            matches.sort(key=lambda item: item[0])
            return [{"text": matches[0][1]}]

        return []
    except Exception:
        return []
