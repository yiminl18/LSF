from __future__ import annotations
import re


_ROLE_RE = re.compile(
    r"(?is)^\s*(?:Hon\.?\s*)?(?P<name>.+?)\s*,\s*"
    r"(?:(?:United States\s+|U\.S\.\s+)?(?:Chief|Senior)\s+)?"
    r"(?:United States\s+|U\.S\.\s+)?District Judge(?:,\s*|\s+)Presiding\s*[\.\*\u2020]*\s*$"
)
_MAGISTRATE_RE = re.compile(
    r"(?is)^\s*(?:Hon\.?\s*)?(?P<name>.+?)\s*,\s*"
    r"(?:United States\s+|U\.S\.\s+)?Magistrate Judge(?:,\s*|\s+)Presiding\s*[\.\*\u2020]*\s*$"
)
_BANKRUPTCY_RE = re.compile(
    r"(?is)^\s*(?:Hon\.?\s*)?(?P<name>.+?)\s*,\s*"
    r"(?:United States\s+|U\.S\.\s+)?Bankruptcy Judges?(?:,\s*|\s+)Presiding\s*[\.\*\u2020]*\s*$"
)

_ROLE_HINT_RE = re.compile(
    r"(?i)\b(?:(?:chief|senior)\s+)?district judge\b|\bmagistrate judge\b|\bbankruptcy judges?\b"
)
_NAMEISH_RE = re.compile(
    r"(?i)^\s*(?:Hon\.?\s*)?[A-Z][A-Za-z0-9.'’\-]*(?:\s+[A-Z][A-Za-z0-9.'’\-]*){0,6}\s*,?\s*$"
)
_NAME_FRAGMENT_RE = re.compile(
    r"(?i)^\s*(?:Hon\.?\s*)?[A-Z][A-Za-z0-9.'' \-]{0,120}(?:,\s*and| and|,|&)\s*$"
)
_NON_NAME_HINT_RE = re.compile(
    r"(?i)\b(?:appeal|appeals|district court|bankruptcy appellate panel|bankruptcy court|"
    r"court of appeals|before:|summary|opinion|argued|submitted|filed|before|"
    r"circuit judge|circuit judges|district judge|magistrate judge|bankruptcy judge|"
    r"appellant|appellee|plaintiff|defendant|panel|court|district\b|for the\b)\b"
)


def rule_presiding_district_judge(doc: dict) -> list[dict]:
    try:
        spans = []
        seen = set()

        def add_name(raw_name: str, source_item: dict) -> None:
            name = re.sub(r"(?i)^\s*Hon\.?\s*", "", raw_name or "").strip()
            name = name.strip(" \t\r\n,.;:*")
            name = re.sub(r"\s+", " ", name).strip()
            if not name:
                return
            key = name.lower()
            if key in seen:
                return
            seen.add(key)
            span = {"text": name}
            for field in ("page_no", "line_no", "paragraph_no"):
                if field in source_item:
                    span[field] = source_item[field]
            spans.append(span)

        def find_match(text: str) -> str | None:
            normalized = re.sub(r"\s+", " ", (text or "")).strip()
            if not normalized:
                return None
            for pat in (_ROLE_RE, _MAGISTRATE_RE, _BANKRUPTCY_RE):
                match = pat.match(normalized)
                if match:
                    return match.group("name")
            return None

        def scan_items(items: list[dict]) -> None:
            limit = min(len(items or []), 120)

            def is_nameish(text: str) -> bool:
                normalized = re.sub(r"\s+", " ", text or "").strip()
                if not normalized or _NON_NAME_HINT_RE.search(normalized):
                    return False
                return bool(_NAMEISH_RE.match(normalized) or _NAME_FRAGMENT_RE.match(normalized))

            for i in range(limit):
                item = items[i] or {}
                text = (item.get("text") or "").strip()
                if not text:
                    continue

                has_role_hint = bool(_ROLE_HINT_RE.search(text))
                has_presiding = "presiding" in text.lower()
                prev_text = ((items[i - 1] or {}).get("text") or "").strip() if i > 0 else ""
                next_text = ((items[i + 1] or {}).get("text") or "").strip() if i + 1 < limit else ""

                direct_name = find_match(text)
                if direct_name and not (has_role_hint or has_presiding):
                    add_name(direct_name, item)
                    continue

                candidates = []

                if prev_text and is_nameish(prev_text):
                    candidates.append((f"{prev_text} {text}", items[i - 1] or item))

                if next_text and is_nameish(next_text):
                    candidates.append((f"{text} {next_text}", item))

                if prev_text and next_text and is_nameish(prev_text):
                    candidates.append((f"{prev_text} {text} {next_text}", items[i - 1] or item))

                if prev_text and next_text and is_nameish(next_text):
                    candidates.append((f"{prev_text} {text} {next_text}", item))

                matched = False
                for candidate_text, candidate_source in candidates:
                    candidate_name = find_match(candidate_text)
                    if candidate_name:
                        add_name(candidate_name, candidate_source)
                        matched = True
                        break

                if matched:
                    continue

                if direct_name:
                    add_name(direct_name, item)

        scan_items(doc.get("lines") or [])
        scan_items(doc.get("paragraphs") or [])

        if len(spans) > 1 and not any(
            any(sep in (span.get("text") or "") for sep in (",", " and ", ";"))
            for span in spans
        ):
            aggregate_text = "; ".join(span.get("text") or "" for span in spans if span.get("text"))
            if aggregate_text:
                aggregate_key = aggregate_text.lower()
                if aggregate_key not in seen:
                    seen.add(aggregate_key)
                    aggregate_span = {"text": aggregate_text}
                    first_span = spans[0]
                    for field in ("page_no", "line_no", "paragraph_no"):
                        if field in first_span:
                            aggregate_span[field] = first_span[field]
                    spans.insert(0, aggregate_span)

        if not spans:
            # Limit to first 5000 chars — full-text search with (.+?) + dotall
            # causes catastrophic backtracking on long strings.
            full_text = re.sub(r"\s+", " ", doc.get("text") or "").strip()[:5000]
            if full_text:
                for pat in (_ROLE_RE, _MAGISTRATE_RE, _BANKRUPTCY_RE):
                    match = pat.search(full_text)
                    if match:
                        add_name(match.group("name"), {})
                        break

        return spans
    except Exception:
        return []
