import re

_IS_PATH = re.compile(
    r"consolidated\s+statement(?:s)?\s+of\s+(?:operations|income|earnings)",
    re.IGNORECASE,
)
_MDA_PATH = re.compile(
    r"(?:management.{0,3}s discussion"
    r"|results of operations"
    r"|discussion and analysis"
    r"|fiscal\s*\d+\s*summary"
    r"|fiscal\s*\d+\s*consolidated results"
    r"|net income attributable)",
    re.IGNORECASE,
)
_PARENT_ATTR = re.compile(
    r"net\s+(?:income|earnings|loss)\s+attributable\s+to\s+(?!non[- ]?controlling)",
    re.IGNORECASE,
)
_SUB_ENTITY = re.compile(
    r"(?:obligor group"
    r"|deed of cross guarantee"
    r"|basis of preparation"
    r"|guarantor subsidiar)",
    re.IGNORECASE,
)
_FULL_INCOME_STATEMENT_MARKER = re.compile(
    r"(?:cost of sales|cost of goods sold|cost of products sold)",
    re.IGNORECASE,
)


def rule_income_statement_table(doc: dict) -> list[dict]:
    """Net-income source spans.

    For filings with a parent-attribution breakdown, return *focused* MD&A summary spans that
    name 'net (income|earnings) attributable to <parent>' — i.e. exclude full income
    statements (those that include 'Cost of sales') and sub-entity statements (Obligor Group,
    Deed of Cross Guarantee). Otherwise fall back to the Consolidated Statements of
    Operations/Income/Earnings table.
    """
    texts = doc.get("texts", []) or []
    mda_spans = []
    for s in texts:
        path_text = (s.get("structure") or {}).get("path_text", "") or ""
        text = s.get("text", "") or ""
        if _SUB_ENTITY.search(path_text) or _SUB_ENTITY.search(text):
            continue
        if _FULL_INCOME_STATEMENT_MARKER.search(text):
            continue
        if _MDA_PATH.search(path_text) and _PARENT_ATTR.search(text):
            mda_spans.append(s)
    if mda_spans:
        return mda_spans

    out = []
    for s in texts:
        if s.get("label") != "table":
            continue
        path_text = (s.get("structure") or {}).get("path_text", "") or ""
        text = s.get("text", "") or ""
        if _SUB_ENTITY.search(path_text):
            continue
        if _IS_PATH.search(path_text) or _IS_PATH.search(text):
            out.append(s)
    return out
