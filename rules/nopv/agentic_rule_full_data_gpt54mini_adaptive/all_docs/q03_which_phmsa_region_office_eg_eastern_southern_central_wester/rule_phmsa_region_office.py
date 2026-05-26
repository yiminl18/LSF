import re


def rule_phmsa_region_office(doc: dict) -> list[dict]:
    try:
        region_specs = [
            (
                "Eastern Region",
                "Eastern",
                [
                    re.compile(r"\bEastern\s+Region\b", re.IGNORECASE),
                    re.compile(r"840\s+Bear\s+Tavern\s+Road", re.IGNORECASE),
                    re.compile(r"West\s+Trenton,\s*NJ", re.IGNORECASE),
                ],
            ),
            (
                "Southern Region",
                "Southern",
                [
                    re.compile(r"\bSouthern\s+Region\b", re.IGNORECASE),
                    re.compile(r"230\s+Peachtree\s+Street(?:\s+N\.?W\.?)?", re.IGNORECASE),
                    re.compile(r"Atlanta,\s*GA\s*30303", re.IGNORECASE),
                ],
            ),
            (
                "Central Region",
                "Central",
                [
                    re.compile(r"\bCentral\s+Region\b", re.IGNORECASE),
                    re.compile(r"901\s+Locust\s+Street", re.IGNORECASE),
                    re.compile(r"Kansas\s+City,\s*MO\s*64106", re.IGNORECASE),
                ],
            ),
            (
                "Western Region",
                "Western",
                [
                    re.compile(r"\bWestern\s+Region\b", re.IGNORECASE),
                    re.compile(r"12300\s+W\.?\s+Dakota\s+Ave\.?", re.IGNORECASE),
                    re.compile(r"Lakewood,\s*CO\s*80228", re.IGNORECASE),
                ],
            ),
            (
                "Southwest Region",
                "Southwest",
                [
                    re.compile(r"\bSouthwest\s+Region\b", re.IGNORECASE),
                    re.compile(r"8701\s+S\.?\s+Gessner", re.IGNORECASE),
                    re.compile(r"Houston,\s*TX\s*77074", re.IGNORECASE),
                    re.compile(r"Houston\s+TX\s*77074", re.IGNORECASE),
                ],
            ),
        ]

        lines = doc.get("lines") or []
        ordered_lines = sorted(
            lines,
            key=lambda x: (
                x.get("page_no", 10**9),
                x.get("line_no", 10**9),
            ),
        )

        def build_spans(region_phrase: str, region_short: str, item: dict | None) -> list[dict]:
            spans = [{"text": region_phrase}]
            if region_short != region_phrase:
                spans.append({"text": region_short})
            if item is not None:
                for span in spans:
                    span["page_no"] = item.get("page_no")
                    span["line_no"] = item.get("line_no")
            return spans

        # Prefer concrete header/footer lines, because they tend to identify the
        # issuing office directly and avoid false positives from body text.
        for item in ordered_lines[:80] + ordered_lines[-80:]:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            for region_phrase, region_short, patterns in region_specs:
                if any(p.search(text) for p in patterns):
                    return build_spans(region_phrase, region_short, item)

        full_text = doc.get("text") or ""
        for region_phrase, region_short, patterns in region_specs:
            if any(p.search(full_text) for p in patterns):
                return build_spans(region_phrase, region_short, None)

        return []
    except Exception:
        return []
