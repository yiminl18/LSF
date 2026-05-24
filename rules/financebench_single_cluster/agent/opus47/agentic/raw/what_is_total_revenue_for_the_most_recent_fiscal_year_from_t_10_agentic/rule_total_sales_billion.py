import re


def rule_total_sales_billion(doc: dict) -> list[dict]:
    """MD&A summary sentence stating consolidated/worldwide/net sales (or revenues) increased/decreased N% to $X billion. Surfaces the human-readable revenue figure (e.g., '$94.9 billion') so the QA answer matches label formatting cleanly."""
    pattern = re.compile(
        r"(worldwide sales|total revenues?|net sales|net revenues?|revenues?)\s+(increased|decreased)\s+[\d.,]+\s*%\s+to\s+\$?[\d.,]+\s*billion",
        re.IGNORECASE,
    )
    out = []
    for span in doc.get("texts", []):
        text = span.get("text", "") or ""
        if pattern.search(text):
            out.append(span)
    return out
