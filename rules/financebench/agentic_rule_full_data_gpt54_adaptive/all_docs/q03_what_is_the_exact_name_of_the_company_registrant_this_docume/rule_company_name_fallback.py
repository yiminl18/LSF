import re


def rule_company_name_fallback(doc: dict) -> list[dict]:
    try:
        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        paragraphs = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]
        lines.sort(
            key=lambda item: (
                item.get("page_no", 10**9),
                item.get("line_no", 10**9),
                item.get("text", ""),
            )
        )
        paragraphs.sort(
            key=lambda item: (
                item.get("page_no", 10**9),
                item.get("paragraph_no", 10**9),
                item.get("text", ""),
            )
        )

        def clean(text: str) -> str:
            return " ".join((text or "").replace("\u00a0", " ").split())

        def make_span(source: dict, text: str) -> list[dict]:
            span = {"text": text}
            for key in ("page_no", "line_no", "paragraph_no"):
                if source.get(key) is not None:
                    span[key] = source.get(key)
            return [span]

        boilerplate_pat = re.compile(
            r"(?i)\b(?:"
            r"table of contents|form\s+10-k|form\s+10-q|form\s+8-k|"
            r"quarterly report|annual report|current report|news release|"
            r"securities and exchange commission|commission file|"
            r"address of principal executive offices|zip code|"
            r"registrant'?s telephone number|trading symbol|"
            r"part\s+[ivx]+|item\s+\d+[a-z]?|"
            r"condensed consolidated|consolidated statements|"
            r"documents incorporated by reference"
            r")\b"
        )
        company_pat = (
            r"[A-Z][A-Za-z0-9&'’.-]*"
            r"(?:[ ,]+[A-Z0-9][A-Za-z0-9&'’.-]*){0,7}"
            r"(?:,?[ ](?:Inc\.?|Incorporated|Corporation|Corp\.?|Company|Co\.|PLC|plc|Ltd\.?|Limited))"
        )

        def extract_name(text: str) -> str | None:
            value = clean(text)
            if not value or len(value) > 180:
                return None
            for pattern in (
                rf"\b(?P<name>{company_pat})\s*(?:\[(?:NYSE|NASDAQ|OTC|AMEX)[^\]]*\]|\((?:NYSE|NASDAQ|OTC|AMEX)[^)]+\))",
                rf"\b(?P<name>{company_pat})['’]s Annual Report on Form\s+10-[KQ]\b",
                rf"^(?P<name>{company_pat})\s*,?\s+(?:incorporated under the laws|and its subsidiaries\b)",
                rf"\b(?P<name>{company_pat})\s*(?:\[(?:NYSE|NASDAQ|OTC|AMEX)[^\]]*\]|\((?:NYSE|NASDAQ|OTC|AMEX)[^)]+\))\s+today reported\b",
            ):
                match = re.search(pattern, value, re.I)
                if match:
                    name = clean(match.group("name")).strip(" ,.;:-")
                    if name and not boilerplate_pat.search(name):
                        return name
            return None

        # Early line scan handles release headlines and ticker-tagged leads.
        for item in lines[:450]:
            text = clean(item.get("text", ""))
            if not text:
                continue
            name = extract_name(text)
            if name:
                return make_span(item, name)

        # Paragraph scan handles OCR where the cover is absent and the first
        # business/introduction paragraph names the registrant.
        for item in paragraphs[:30]:
            text = clean(item.get("text", ""))
            if not text or boilerplate_pat.search(text):
                continue
            name = extract_name(text)
            if name:
                return make_span(item, name)

        return []
    except Exception:
        return []
