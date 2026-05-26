import re


_ACTIVITY_RE = re.compile(
    r"\b(?:hazardous\s+liquid|liquefied\s+natural\s+gas|LNG(?:\s+facility|\s+plant)?|"
    r"drug\s+and\s+alcohol|anti-drug|alcohol\s+misuse|covered\s+function|covered\s+employee|"
    r"D&A|gas\s+transmission|gas\s+distribution|distribution\s+system|"
    r"transmission\s+pipeline|transmission\s+line|gathering\s+pipeline|gathering\s+system|"
    r"underground\s+natural\s+gas\s+storage|annual\s+report|incident\s+report)\b",
    re.IGNORECASE,
)
_INTRO_RE = re.compile(
    r"^\s*From\b|\b(?:inspected|inspection|reviewed|audit(?:ed)?|investigat(?:ed|ion))\b",
    re.IGNORECASE,
)
_PART_RE = re.compile(
    r"\b(?:49\s*C\.?\s*F\.?\s*R\.?\s*)?(?:Part|part)\s*(191|192|193|195|199)\b",
    re.IGNORECASE,
)
_STOP_RE = re.compile(
    r"^\s*(?:\d+\.\s*$|proposed\s+civil\s+penalty|proposed\s+compliance\s+order|"
    r"response\s+to\s+this\s+notice|warning\s+items|sincerely|respectfully)\b",
    re.IGNORECASE,
)


def rule_operator_activity_context(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict) and norm(item.get("text") or "")]
        if not lines:
            return []

        def collect(start: int, max_lines: int) -> tuple[str, dict]:
            page_no = lines[start].get("page_no")
            parts = []
            for idx in range(start, min(len(lines), start + max_lines)):
                item = lines[idx]
                text = norm(item.get("text") or "")
                if not text:
                    continue
                if idx > start and item.get("page_no") != page_no:
                    break
                if idx > start and _STOP_RE.match(text):
                    break
                if idx > start and re.match(r"^\s*\d+\.\s*$", text):
                    break
                parts.append(text)
            return "\n".join(parts), lines[start]

        candidates = []
        seen = set()

        for i, item in enumerate(lines[:120]):
            text = norm(item.get("text") or "")
            if not text:
                continue

            if re.match(r"^\s*From\b", text, re.IGNORECASE):
                snippet, source = collect(i, 6)
                if snippet and _ACTIVITY_RE.search(snippet):
                    key = snippet.lower()
                    if key not in seen:
                        seen.add(key)
                        span = {"text": snippet}
                        for field in ("page_no", "line_no"):
                            value = source.get(field)
                            if value is not None:
                                span[field] = value
                        candidates.append((0, i, span))

            if _PART_RE.search(text):
                snippet, source = collect(i, 3)
                if snippet and _ACTIVITY_RE.search(snippet):
                    key = snippet.lower()
                    if key not in seen:
                        seen.add(key)
                        span = {"text": snippet}
                        for field in ("page_no", "line_no"):
                            value = source.get(field)
                            if value is not None:
                                span[field] = value
                        candidates.append((1, i, span))

            if (not re.match(r"^\s*From\b", text, re.IGNORECASE)) and (not _PART_RE.search(text)) and _INTRO_RE.search(text):
                snippet, source = collect(i, 4)
                if snippet and _ACTIVITY_RE.search(snippet):
                    key = snippet.lower()
                    if key not in seen:
                        seen.add(key)
                        span = {"text": snippet}
                        for field in ("page_no", "line_no"):
                            value = source.get(field)
                            if value is not None:
                                span[field] = value
                        candidates.append((2, i, span))

        candidates.sort()
        return [item[-1] for item in candidates[:2]]
    except Exception:
        return []
