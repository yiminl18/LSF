import re


_EXPLICIT_PART_RE = re.compile(
    r"\b(?:49\s*C\.?\s*F\.?\s*R\.?\s*)?(?:Part|part)\s*(191|192|193|195|199)\b",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(
    r"(?<!\d)(?:49\s*C\.?\s*F\.?\s*R\.?\s*)?(?:§\s*)?(191|192|193|195|199)\.\d{1,4}(?:\([a-z0-9]+\))*",
    re.IGNORECASE,
)
_COVERAGE_RE = re.compile(
    r"\b(?:subject\s+to|covered\s+by|regulated\s+under|regulated\s+by|"
    r"requirements?\s+of|in\s+accordance\s+with|required\s+by|applies\s+to|"
    r"this\s+part|this\s+subpart|each\s+operator|must\s+meet)\b",
    re.IGNORECASE,
)
_ACTIVITY_RE = re.compile(
    r"\b(?:hazardous\s+liquid|liquefied\s+natural\s+gas|LNG(?:\s+facility|\s+plant)?|"
    r"drug\s+and\s+alcohol|anti-drug|alcohol\s+misuse|covered\s+function|covered\s+employee|"
    r"D&A|gas\s+transmission|gas\s+distribution|distribution\s+system|transmission\s+pipeline|"
    r"transmission\s+line|gathering\s+pipeline|gathering\s+system|underground\s+natural\s+gas\s+storage|"
    r"annual\s+report|incident\s+report)\b",
    re.IGNORECASE,
)
_STOP_RE = re.compile(
    r"^\s*(?:proposed\s+civil\s+penalty|proposed\s+compliance\s+order|response\s+to\s+this\s+notice|"
    r"warning\s+items|sincerely|respectfully)\b",
    re.IGNORECASE,
)


def rule_regulated_part_citations(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def part_priority(parts: set[str]) -> int:
            if any(part in {"192", "193", "195", "199"} for part in parts):
                return 0
            if "191" in parts:
                return 1
            return 2

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict) and norm(item.get("text") or "")]
        if not lines:
            return []

        def collect(start: int, before: int, after: int) -> tuple[str, dict]:
            first = max(0, start - before)
            page_no = lines[start].get("page_no")
            parts = []
            source = lines[start]
            for idx in range(first, min(len(lines), start + after + 1)):
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
            return "\n".join(parts), source

        candidates = []
        seen = set()

        saw_relevant = False

        for i, item in enumerate(lines):
            text = norm(item.get("text") or "")
            if not text:
                continue
            if saw_relevant and _STOP_RE.match(text):
                break

            explicit_parts = set(_EXPLICIT_PART_RE.findall(text))
            section_parts = set(_SECTION_RE.findall(text))
            parts = explicit_parts | section_parts
            if not parts:
                continue
            saw_relevant = True

            if explicit_parts and not section_parts:
                snippet, source = collect(i, 0, 1)
            elif section_parts:
                snippet, source = collect(i, 0, 4)
            else:
                snippet, source = collect(i, 0, 2)

            compact = norm(snippet)
            if not compact:
                continue

            has_activity = bool(_ACTIVITY_RE.search(compact))
            has_coverage = bool(_COVERAGE_RE.search(compact))
            starts_like_heading = text.startswith("§") or re.match(r"^\s*\d+\.\s*§", text)

            rank = 6
            if explicit_parts and part_priority(parts) == 0 and (has_activity or has_coverage):
                rank = 0
            elif explicit_parts and part_priority(parts) == 0:
                rank = 1
            elif section_parts and part_priority(parts) == 0 and (starts_like_heading or has_activity or has_coverage):
                rank = 2
            elif explicit_parts and "191" in parts and (has_activity or has_coverage):
                rank = 3
            elif section_parts and "191" in parts and (starts_like_heading or has_activity or has_coverage):
                rank = 4
            elif explicit_parts or section_parts:
                rank = 5

            key = compact.lower()
            if key in seen:
                continue
            seen.add(key)

            span = {"text": snippet}
            for field in ("page_no", "line_no"):
                value = source.get(field)
                if value is not None:
                    span[field] = value

            candidates.append((rank, part_priority(parts), i, span))

        candidates.sort()
        return [item[-1] for item in candidates[:4]]
    except Exception:
        return []
