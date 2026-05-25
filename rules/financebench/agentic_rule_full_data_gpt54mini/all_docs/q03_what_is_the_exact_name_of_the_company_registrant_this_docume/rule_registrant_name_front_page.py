import re


_LABEL_RE = re.compile(r"exact name of registrant(?:s)?\b", re.IGNORECASE)
_STOP_RE = re.compile(
    r"\b(?:commission file number|form\s+10-|form\s+8-|united states|"
    r"securities and exchange commission|washington,?\s*d\.?c\.?|"
    r"pursuant to section|date of report|date of the report|"
    r"address of principal executive offices|registrant's telephone number|"
    r"registrant’s telephone number|table of contents)\b",
    re.IGNORECASE,
)
_JURISDICTION_RE = re.compile(
    r"\b(?:delaware|california|new york|north carolina|texas|florida|"
    r"nevada|illinois|pennsylvania|massachusetts|maryland|virginia|"
    r"ohio|minnesota|indiana|michigan|oregon|georgia|washington)\b",
    re.IGNORECASE,
)
_COMPANY_LINE_RE = re.compile(
    r"^(?:[A-Z][A-Za-z0-9&.'’/-]*|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9&.'’/-]*|[A-Z]{2,})){0,6},?\s+"
    r"(?:Inc\.?|Incorporated|Corporation|Corp\.?|Company|plc|PLC|LLC|L\.L\.C\.|"
    r"Ltd\.?|Limited Partnership|LP)$",
)
_COMPANY_TEXT_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&.'’/-]*(?:\s+[A-Z][A-Za-z0-9&.'’/-]*){0,6},?\s+"
    r"(?:Inc\.?|Incorporated|Corporation|Corp\.?|Company|plc|PLC|LLC|L\.L\.C\.|"
    r"Ltd\.?|Limited Partnership|LP))\b",
)


def rule_registrant_name_front_page(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        items = [s for s in (doc.get("lines") or []) if isinstance(s, dict)]
        if not items:
            items = [s for s in (doc.get("paragraphs") or []) if isinstance(s, dict)]
        if not items:
            items = [s for s in (doc.get("pages") or []) if isinstance(s, dict)]
        if not items:
            return []

        texts = [norm(item.get("text") or "") for item in items]
        joined_text = norm(doc.get("text") or "")

        def make_span(idx: int, text: str) -> dict:
            span = {"text": text}
            for key in ("page_no", "line_no", "paragraph_no"):
                if key in items[idx]:
                    span[key] = items[idx][key]
            return span

        def company_span_from_text(text: str) -> dict | None:
            candidate = norm(text)
            if not candidate or len(candidate) > 140:
                return None
            if not any(ch.isalpha() for ch in candidate):
                return None
            if not _COMPANY_LINE_RE.match(candidate):
                return None
            if _STOP_RE.search(candidate) or _JURISDICTION_RE.search(candidate):
                return None
            return {"text": candidate}

        def company_span_at(idx: int) -> dict | None:
            candidate = company_span_from_text(texts[idx])
            if not candidate:
                return None
            for key in ("page_no", "line_no", "paragraph_no"):
                if key in items[idx]:
                    candidate[key] = items[idx][key]
            return candidate

        for idx, text in enumerate(texts):
            if not text or not _LABEL_RE.search(text):
                # Handle OCR splits such as "(Exact" followed by "name of registrant ...".
                if not text or "exact" not in text.lower():
                    continue
                next_idx = idx + 1
                if next_idx < len(texts) and "name of registrant" in texts[next_idx].lower():
                    # Treat the second line as the label anchor.
                    idx = next_idx
                    text = texts[idx]
                else:
                    continue

            spans: list[dict] = []

            # Capture contiguous name lines immediately above the registrant label.
            back = idx - 1
            while back >= 0:
                prev_text = texts[back]
                if not prev_text:
                    break
                if _STOP_RE.search(prev_text):
                    break
                if _LABEL_RE.search(prev_text):
                    break
                spans.append(make_span(back, prev_text))
                back -= 1

            spans.reverse()

            # Multi-registrant filings sometimes repeat the first registrant after the label.
            # Include that line when it looks like another company name and not a jurisdiction.
            forward = idx + 1
            while forward < len(texts) and not texts[forward]:
                forward += 1
            if forward < len(texts):
                next_text = texts[forward]
                if (
                    next_text
                    and not _STOP_RE.search(next_text)
                    and not _JURISDICTION_RE.search(next_text)
                    and len(next_text) <= 120
                    and any(ch.isalpha() for ch in next_text)
                ):
                    if next_text not in {span["text"] for span in spans}:
                        spans.append(make_span(forward, next_text))

            if spans:
                return spans

            # Fallback for OCR that keeps the company name on the same line as the label.
            same_line_prefix = text.split("(", 1)[0].strip(" :-\u00a0\t")
            if same_line_prefix:
                return [make_span(idx, same_line_prefix)]

        # Fallback for OCR cases where the front-page label is missing or heavily mangled.
        for idx, text in enumerate(texts):
            candidate = company_span_at(idx)
            if candidate:
                return [candidate]

        if joined_text:
            # Search the raw document text as a last resort.
            match = _COMPANY_TEXT_RE.search(joined_text)
            if match:
                return [{"text": norm(match.group(1))}]

        return []
    except Exception:
        return []
