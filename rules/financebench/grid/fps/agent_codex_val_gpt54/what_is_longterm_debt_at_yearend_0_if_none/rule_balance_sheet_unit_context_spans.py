def rule_balance_sheet_unit_context_spans(doc: dict) -> list[dict]:
    """Match nearby unit and heading spans immediately before debt-bearing balance tables."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            path = (((span.get("structure") or {}).get("path_text") or "")).lower()
            text = (span.get("text") or "").lower()
            if (
                span.get("label") == "table"
                and any(
                    key in path
                    for key in ("balance sheet", "balance sheets", "financial position", "capitalization")
                )
                and "unaudited" not in path
                and any(
                    key in text
                    for key in (
                        "long-term debt",
                        "long term debt",
                        "term debt",
                        "current maturities of long-term debt",
                        "obligations under finance leases",
                    )
                )
            ):
                for j in (i - 2, i - 1):
                    if j < 0:
                        continue
                    prev = texts[j]
                    prev_text = (prev.get("text") or "").lower()
                    prev_path = (((prev.get("structure") or {}).get("path_text") or "")).lower()
                    if (
                        any(
                            cue in prev_text
                            for cue in ("in thousands", "in millions", "in billions", "amounts in millions")
                        )
                        or (
                            prev.get("label") == "section_header"
                            and any(
                                cue in prev_text
                                for cue in ("balance sheet", "balance sheets", "financial position", "capitalization")
                            )
                        )
                        or (
                            prev.get("label") == "section_header"
                            and prev_path == path
                        )
                    ):
                        out.append(prev)
        return out
    except Exception:
        return []
