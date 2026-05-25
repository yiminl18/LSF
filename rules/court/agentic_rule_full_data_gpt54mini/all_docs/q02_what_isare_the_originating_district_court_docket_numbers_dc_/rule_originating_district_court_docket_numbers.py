import re


_LABEL_RE = re.compile(r"(?i)\bD\.?\s*C\.?\s*No?s?\.?\b")
_FULL_DOCKET_RE = re.compile(
    r"\b(?:\d{1,2}\s*:\s*\d{2}\s*-?\s*|\d{1,2}\s*-?\s*)"
    r"(?:cv|cr|mc|md)\s*-?\s*"
    r"\d{1,6}(?:\s*[A-Z]{1,4})?(?:\s*-\s*[A-Z0-9]{1,6}){0,3}\b",
    re.IGNORECASE,
)
_PREFIX_RE = re.compile(r"\b\d{1,2}\s*:\s*\d{2}\s*-?\s*(?:cv|cr|mc|md)", re.IGNORECASE)
_SUFFIX_ONLY_RE = re.compile(r"^\s*\d{1,6}(?:\s*-\s*[A-Z0-9]{1,6}){0,3}\s*$", re.IGNORECASE)


def rule_originating_district_court_docket_numbers(doc: dict) -> list[dict]:
    try:
        spans = []
        seen = set()

        def add_span(text: str, source_item: dict) -> None:
            cleaned = re.sub(r"\s+", "", text or "").rstrip(".,;:")
            if not cleaned:
                return
            key = cleaned.lower()
            if key in seen:
                return
            seen.add(key)
            span = {"text": cleaned}
            for field in ("page_no", "line_no", "paragraph_no"):
                if field in source_item:
                    span[field] = source_item[field]
            spans.append(span)

        def scan_items(items: list[dict], max_items: int = 120) -> None:
            limit = min(len(items or []), max_items)
            for i in range(limit):
                item = items[i] or {}
                text = (item.get("text") or "").strip()
                if not text:
                    continue

                label_match = _LABEL_RE.search(text)
                if not label_match and not text.upper().startswith("D.C."):
                    continue

                remainder = text[label_match.end():].strip() if label_match else text

                direct_matches = list(_FULL_DOCKET_RE.finditer(remainder))
                if direct_matches:
                    for match in direct_matches:
                        add_span(match.group(0), item)
                    continue

                prefix_match = _PREFIX_RE.search(remainder)
                if prefix_match:
                    prefix = re.sub(r"\s+", "", prefix_match.group(0))
                    after_prefix = remainder[prefix_match.end():].strip()

                    if _SUFFIX_ONLY_RE.fullmatch(after_prefix):
                        add_span(prefix + re.sub(r"\s+", "", after_prefix), item)
                        continue

                    for j in range(i + 1, min(limit, i + 5)):
                        next_item = items[j] or {}
                        next_text = (next_item.get("text") or "").strip()
                        if not next_text:
                            continue

                        next_direct_matches = list(_FULL_DOCKET_RE.finditer(next_text))
                        if next_direct_matches:
                            for match in next_direct_matches:
                                add_span(match.group(0), next_item)
                            continue

                        if _SUFFIX_ONLY_RE.fullmatch(next_text):
                            add_span(prefix + re.sub(r"\s+", "", next_text), next_item)
                            break

                        if not _LABEL_RE.search(next_text):
                            break

                elif label_match:
                    for j in range(i + 1, min(limit, i + 5)):
                        next_item = items[j] or {}
                        next_text = (next_item.get("text") or "").strip()
                        if not next_text:
                            continue

                        next_direct_matches = list(_FULL_DOCKET_RE.finditer(next_text))
                        if next_direct_matches:
                            for match in next_direct_matches:
                                add_span(match.group(0), next_item)
                            continue

                        if _SUFFIX_ONLY_RE.fullmatch(next_text):
                            # This can happen when OCR splits the docket number across lines.
                            prev_prefix = _PREFIX_RE.search(text)
                            if prev_prefix:
                                add_span(
                                    re.sub(r"\s+", "", prev_prefix.group(0))
                                    + re.sub(r"\s+", "", next_text),
                                    next_item,
                                )
                            break

                        if not _LABEL_RE.search(next_text):
                            break

        scan_items(doc.get("lines") or [])
        scan_items(doc.get("paragraphs") or [])

        if not spans:
            full_text = doc.get("text") or ""
            for match in re.finditer(r"(?i)D\.?\s*C\.?\s*No?s?\.?\s*([^\n]{0,120})", full_text):
                chunk = match.group(1) or ""
                for docket_match in _FULL_DOCKET_RE.finditer(chunk):
                    add_span(docket_match.group(0), {})

        return spans
    except Exception:
        return []
