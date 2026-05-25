import re


def rule_cover_page_security_listing(doc: dict) -> list[dict]:
    '''Page-1 cover spans naming the trading symbol(s), share class, or listing exchange; also grabs the few spans following a "Symbol" / "Trading Symbol(s)" label so the ticker value is included.'''
    pat = re.compile(
        r'(trading\s+symbol'
        r'|name\s+of\s+each\s+exchange'
        r'|exchange\s+on\s+which\s+registered'
        r'|common\s+stock'
        r'|ordinary\s+shares'
        r'|preferred\s+stock'
        r'|nasdaq'
        r'|new\s+york\s+stock\s+exchange'
        r'|\bnyse\b'
        r'|title\s+of\s+each\s+class)',
        re.I,
    )
    sym_label = re.compile(r'^\s*\(?\s*(?:trading\s+)?symbol\s*\(?s?\)?\s*\)?\s*$', re.I)
    texts = doc.get("texts", [])
    keep = set()
    for i, s in enumerate(texts):
        if s.get("page_no") != 1:
            continue
        t = s.get("text") or ""
        if pat.search(t):
            keep.add(i)
        if sym_label.match(t):
            keep.add(i)
            for j in range(i + 1, min(i + 6, len(texts))):
                if texts[j].get("page_no") == 1:
                    keep.add(j)
                else:
                    break
    return [texts[i] for i in sorted(keep)]
