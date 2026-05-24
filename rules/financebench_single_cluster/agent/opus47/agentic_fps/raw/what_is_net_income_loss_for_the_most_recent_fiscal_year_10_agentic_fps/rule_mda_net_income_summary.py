import re


def rule_mda_net_income_summary(doc: dict) -> list[dict]:
    """MD&A short summary tables headlined '(Dollars In Billions)' that include a Net Earnings/Income label.

    Targets cash-flow / liquidity executive summaries that quote the headline net
    earnings figure in billion-scale units (e.g. J&J's MD&A table '$ 17.9 Net Earnings').
    """
    mda_pat = re.compile(
        r"(?:management.{0,3}s discussion|results of operations|discussion and analysis)",
        re.IGNORECASE,
    )
    dollars_billions = re.compile(r"\(?\s*dollars\s+in\s+billions\s*\)?", re.IGNORECASE)
    net_label = re.compile(r"net\s+(?:earnings|income|loss)", re.IGNORECASE)
    out = []
    for span in doc.get("texts", []):
        if span.get("label") != "table":
            continue
        path_text = (span.get("structure") or {}).get("path_text", "") or ""
        text = span.get("text", "") or ""
        if mda_pat.search(path_text) and dollars_billions.search(text) and net_label.search(text):
            out.append(span)
    return out
