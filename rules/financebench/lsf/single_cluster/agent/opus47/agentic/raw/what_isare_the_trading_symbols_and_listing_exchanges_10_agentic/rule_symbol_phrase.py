import re


def rule_symbol_phrase(doc: dict) -> list[dict]:
    """Spans that explicitly state the trading/ticker symbol via 'under the symbol X' phrasing."""
    pat = re.compile(
        r"(under the symbol|ticker symbol|trading symbol|"
        r"trades (?:on|under)|listed on .* under|"
        r"symbol\s+[\"“]?[A-Z]{1,6}[\"”]?)",
        re.IGNORECASE,
    )
    keep = []
    for span in doc.get("texts", []):
        text = (span.get("text") or "").strip()
        if not text:
            continue
        if pat.search(text):
            keep.append(span)
    return keep
