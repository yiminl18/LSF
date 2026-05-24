import re


def rule_cover_page_trading_symbol_exchange(doc: dict) -> list[dict]:
    """Page-1 spans naming the registered class, trading symbol, and listing exchange."""
    content_re = re.compile(
        r"\b(Trading Symbol|Title of each class|Title of Each Class|"
        r"Name of each exchange|Name of Each Exchange|under the symbol|"
        r"ticker symbol|NASDAQ|Nasdaq|New York Stock Exchange|NYSE|"
        r"Stock Market|Common Stock|Common stock|Ordinary Share|Class [ABC] Common)",
        re.IGNORECASE,
    )
    ticker_re = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,5}$")
    keep = []
    for span in doc.get("texts", []):
        if span.get("page_no") != 1:
            continue
        text = (span.get("text") or "").strip()
        if not text:
            continue
        if content_re.search(text):
            keep.append(span)
            continue
        path = (span.get("structure") or {}).get("path_text", "") or ""
        is_short_token = ticker_re.match(text) and 2 <= len(text) <= 6
        if is_short_token and (
            "Trading Symbol" in path or span.get("all_cap") == 1 or span.get("bold") == 1
        ):
            keep.append(span)
    return keep
