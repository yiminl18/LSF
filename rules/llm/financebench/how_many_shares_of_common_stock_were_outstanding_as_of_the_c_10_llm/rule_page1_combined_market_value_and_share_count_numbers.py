def rule_page1_combined_market_value_and_share_count_numbers(doc: dict) -> list[dict]:
    """Match spans containing two large numbers where one corresponds to market value and one to share count."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "")
            nums = re.findall(r"\d[\d,]{5,}", text)
            t = text.lower()
            if len(nums) >= 2 and ("market value" in t or "number of shares of common stock outstanding" in t):
                out.append(span)
        return out
    except Exception:
        return []
