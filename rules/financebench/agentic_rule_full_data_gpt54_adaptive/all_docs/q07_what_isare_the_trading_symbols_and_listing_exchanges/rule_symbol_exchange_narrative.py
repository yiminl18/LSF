import re


_KEY_PHRASE_RE = re.compile(
    r"\[\s*(NYSE|Nasdaq)\s*:\s*[A-Z][A-Z0-9.\-]{0,12}\s*\]"
    r"|ticker\s+symbol"
    r"|stock\s+ticker\s+symbol"
    r"|under\s+the\s+symbol"
    r"|under\s+the\s+ticker\s+symbol"
    r"|trades\s+under\s+the\s+symbol"
    r"|quoted\s+on\s+.+?\s+under\s+the\s+symbol"
    r"|traded\s+on\s+.+?\s+under\s+the\s+ticker\s+symbol",
    re.IGNORECASE,
)
_EXCHANGE_RE = re.compile(
    r"\b(?:"
    r"New\s+York\s+Stock\s+Exchange(?:,\s*Inc\.)?"
    r"|Chicago\s+Stock\s+Exchange(?:,\s*Inc\.)?"
    r"|The\s+Nasdaq\s+Stock\s+Market\s+LLC"
    r"|The\s+Nasdaq\s+Global\s+Select\s+Market"
    r"|Nasdaq(?:\s+Global\s+Select\s+Market|\s+Global\s+Market|\s+Capital\s+Market|\s+National\s+Market)?"
    r"|NASDAQ(?:\s+Global\s+Select\s+Market|\s+Global\s+Market|\s+Capital\s+Market|\s+National\s+Market)?"
    r"|NYSE(?:\s+American|\s+Arca)?"
    r")\b",
    re.IGNORECASE,
)
_SYMBOL_HINT_RE = re.compile(
    r"\[\s*(?:NYSE|Nasdaq)\s*:\s*[A-Z][A-Z0-9.\-]{0,12}\s*\]"
    r"|symbol\s+[\"“']?[A-Z][A-Z0-9.\-]{0,12}[\"”']?"
    r"|ticker\s+symbol\s+(?:is\s+)?[\"“']?[A-Z][A-Z0-9.\-]{0,12}[\"”']?",
    re.IGNORECASE,
)
_NOISE_RE = re.compile(
    r"national\s+securities\s+exchange|recognized\s+stock\s+exchange|last\s+trading\s+day",
    re.IGNORECASE,
)


def rule_symbol_exchange_narrative(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            text = (text or "").replace("\u00a0", " ")
            return re.sub(r"\s+", " ", text).strip()

        def extract_snippet(text: str) -> str:
            cleaned = norm(text)
            if not cleaned:
                return ""

            match = _KEY_PHRASE_RE.search(cleaned)
            if not match:
                return ""

            start = match.start()
            left_window = cleaned[max(0, start - 180) : start]
            sentence_start = max(
                left_window.rfind(". "),
                left_window.rfind("; "),
                left_window.rfind(": "),
            )
            if sentence_start >= 0:
                start = max(0, start - len(left_window) + sentence_start + 2)
            else:
                stock_match = re.search(
                    r"the\s+principal\s+market\s+for\s+our\s+common\s+stock"
                    r"|the\s+company'?s\s+common\s+stock"
                    r"|our\s+common\s+stock"
                    r"|[A-Z][A-Za-z&.,' ]+\[(?:NYSE|Nasdaq)",
                    cleaned[max(0, start - 180) :],
                    re.IGNORECASE,
                )
                if stock_match:
                    start = max(0, start - 180 + stock_match.start())
                else:
                    start = max(0, start - 80)

            end = min(len(cleaned), match.end() + 180)
            tail = cleaned[match.end() : end]
            sentence_end_candidates = [pos for pos in (tail.find(". "), tail.find("; ")) if pos >= 0]
            if sentence_end_candidates:
                end = match.end() + min(sentence_end_candidates) + 1

            return cleaned[start:end].strip(" -")

        def good_candidate(text: str) -> bool:
            cleaned = norm(text)
            if not cleaned:
                return False
            if _NOISE_RE.search(cleaned):
                return False
            if not _KEY_PHRASE_RE.search(cleaned):
                return False
            return bool(_EXCHANGE_RE.search(cleaned) or _SYMBOL_HINT_RE.search(cleaned))

        def dedupe(spans: list[dict]) -> list[dict]:
            seen: set[str] = set()
            out: list[dict] = []
            for span in spans:
                key = norm(span.get("text") or "").lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(span)
            return out

        spans: list[dict] = []

        for item in doc.get("lines") or []:
            if not isinstance(item, dict):
                continue
            text = norm(item.get("text") or "")
            if not good_candidate(text):
                continue
            snippet = extract_snippet(text) or text
            span = {"text": snippet}
            if item.get("page_no") is not None:
                span["page_no"] = item.get("page_no")
            if item.get("line_no") is not None:
                span["line_no"] = item.get("line_no")
            spans.append(span)

        if spans:
            return dedupe(spans[:3])

        for item in doc.get("paragraphs") or []:
            if not isinstance(item, dict):
                continue
            text = norm(item.get("text") or "")
            if not good_candidate(text):
                continue
            snippet = extract_snippet(text) or text
            span = {"text": snippet}
            if item.get("page_no") is not None:
                span["page_no"] = item.get("page_no")
            if item.get("paragraph_no") is not None:
                span["paragraph_no"] = item.get("paragraph_no")
            spans.append(span)

        return dedupe(spans[:3])
    except Exception:
        return []
