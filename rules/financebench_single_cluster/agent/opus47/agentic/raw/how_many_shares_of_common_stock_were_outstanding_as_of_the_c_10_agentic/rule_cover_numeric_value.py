import re

_NUM_PATTERN = re.compile(
    r"^[\$\s]*\d{1,3}(?:,\d{3}){2,}(?:\.\d+)?(?:\s+\d{1,3}(?:,\d{3}){2,}(?:\.\d+)?)*\s*$"
)
_NUM_FINDALL = re.compile(r"\d{1,3}(?:,\d{3}){2,}")


def rule_cover_numeric_value(doc: dict) -> list[dict]:
    '''Cover-page (pages 1-3) numeric-only spans containing at least one value <= 5,000,000,000 — share counts (millions to a few billion shares) that live in a separate span from their label, excluding aggregate-market-value totals.'''
    out = []
    for span in doc.get("texts", []):
        if 1 <= span.get("page_no", 0) <= 3:
            text = span.get("text", "").strip()
            if not _NUM_PATTERN.match(text):
                continue
            vals = [int(n.replace(",", "")) for n in _NUM_FINDALL.findall(text)]
            if vals and any(v <= 5_000_000_000 for v in vals):
                out.append(span)
    return out
