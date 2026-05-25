import re


def rule_traded_under_symbol(doc: dict) -> list[dict]:
    '''Item 5 sentence "traded/listed on <exchange> under the (ticker) symbol <X>"; only fires when the cover page lacks a "Trading symbol" column (older 10-K style).'''
    cover_has_trading = False
    for s in doc.get("texts", []):
        if s.get("page_no") == 1:
            if re.search(r'trading\s+symbol', s.get("text") or "", re.I):
                cover_has_trading = True
                break
    if cover_has_trading:
        return []
    pat = re.compile(
        r'(traded|listed|trades)\s+on\b[^\n]{0,160}\bunder\s+the\s+(?:ticker\s+)?symbol\b',
        re.I,
    )
    return [s for s in doc.get("texts", []) if pat.search(s.get("text") or "")]
