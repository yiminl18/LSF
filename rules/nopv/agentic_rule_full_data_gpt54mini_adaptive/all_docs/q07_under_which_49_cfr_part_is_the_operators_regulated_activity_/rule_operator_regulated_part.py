import re


_EXPLICIT_PART_RE = re.compile(
    r"\b(?:49\s*C\.?\s*F\.?\s*R\.?\s*)?(?:Part|part)\s*(192|195|199)\b",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(
    r"(?<!\d)(?:49\s*C\.?\s*F\.?\s*R\.?\s*)?(?:§\s*)?(192|195|199)\.\d{1,3}(?:\([a-z0-9]+\))*",
    re.IGNORECASE,
)
_COVERAGE_RE = re.compile(
    r"\b(?:subject to|covered by|regulated under|regulated by|requirements? of|in accordance with|"
    r"conforms? to|required by|applies to|this section applies|this part|each operator|"
    r"pipelines? regulated under this part|not subject to)\b",
    re.IGNORECASE,
)


def rule_operator_regulated_part(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def source_rank(source: str) -> int:
            if source == "lines":
                return 0
            if source == "paragraphs":
                return 1
            return 2

        items = []
        for source in ("lines", "paragraphs", "pages"):
            for item in doc.get(source, []) or []:
                if not isinstance(item, dict):
                    continue
                text = norm(item.get("text") or "")
                if text:
                    items.append((source, item, text))

        if not items:
            return []

        candidates = []
        seen_texts = set()

        for source, item, text in items:
            lowered = text.lower()
            explicit = bool(_EXPLICIT_PART_RE.search(text))
            section = bool(_SECTION_RE.search(text))
            coverage = bool(_COVERAGE_RE.search(text))

            if not explicit and not section:
                continue
            if not (coverage or explicit or section):
                continue

            # Prefer short, direct citations or headings that name the part.
            if explicit:
                kind_rank = 0
            elif section and (text.startswith("§") or text.startswith("49 CFR") or "what general requirements" in lowered):
                kind_rank = 1
            elif section and coverage:
                kind_rank = 2
            else:
                kind_rank = 3

            span = {"text": text}
            for field in ("page_no", "paragraph_no", "line_no"):
                if field in item:
                    span[field] = item[field]

            key = norm(span["text"]).lower()
            if key in seen_texts:
                continue
            seen_texts.add(key)

            candidates.append(
                (
                    kind_rank,
                    source_rank(source),
                    span.get("page_no", 0),
                    span.get("paragraph_no", 0),
                    span.get("line_no", 0),
                    span,
                )
            )

        if candidates:
            candidates.sort()
            return [candidates[0][-1]]

        # Fallback: capture a compact snippet around the first relevant citation.
        full_text = doc.get("text") or ""
        for match in re.finditer(r"(?<!\d)(?:49\s*C\.?\s*F\.?\s*R\.?\s*)?(?:Part|part)?\s*(192|195|199)\.\d{1,3}(?:\([a-z0-9]+\))*", full_text, re.I):
            start = max(0, match.start() - 160)
            end = min(len(full_text), match.end() + 220)
            snippet = norm(full_text[start:end])
            if snippet:
                return [{"text": snippet}]

        for match in _EXPLICIT_PART_RE.finditer(full_text):
            start = max(0, match.start() - 160)
            end = min(len(full_text), match.end() + 220)
            snippet = norm(full_text[start:end])
            if snippet:
                return [{"text": snippet}]

        return []
    except Exception:
        return []
