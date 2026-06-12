def rule_esf_exchange_stabilization_fund_or_international_statistics_boundary(doc: dict) -> list[dict]:
    """Match tables between International Statistics and Special Reports that mention ESF or balance sheet."""
    import re
    try:
        texts = doc.get("texts", [])
        in_block = False
        out = []
        for s in texts:
            if s.get("label") == "section_header" and re.search(r'INTERNATIONAL STATISTICS|INTERNATIONAL FINANCIAL STATISTICS', s.get("text", ""), re.I):
                in_block = True
            elif in_block and s.get("label") == "section_header" and re.search(r'SPECIAL REPORTS|TRUST FUND', s.get("text", ""), re.I):
                in_block = False
            if in_block and s.get("label") == "table":
                txt = s.get("text", "") or ""
                path = ((s.get("structure") or {}).get("path_text") or "")
                if re.search(r'ESF|Exchange Stabilization Fund|Balance sheet|Total assets', path + " " + txt, re.I):
                    out.append(s)
        return out
    except Exception:
        return []
