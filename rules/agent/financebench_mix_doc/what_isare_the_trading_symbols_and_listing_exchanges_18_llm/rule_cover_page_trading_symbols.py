def rule_cover_page_trading_symbols(doc: dict) -> list[dict]:
    """Retrieve page-1 cover-page spans around Section 12(b) trading symbol and exchange disclosures."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            try:
                if span.get("page_no") != 1:
                    continue
                label = span.get("label")
                if label not in {"text", "section_header", "table"}:
                    continue
                txt = (span.get("text") or "")
                path = ((span.get("structure") or {}).get("path_text") or "")
                hay = f"{path} {txt}".lower()
                if (
                    "securities registered pursuant to section 12(b)" in hay
                    or "trading symbol" in hay
                    or "trading symbol(s)" in hay
                    or "name of each exchange" in hay
                    or "name of each exchange on which registered" in hay
                    or "exchange on which registered" in hay
                ):
                    out.append(span)
                    continue
                if label == "table":
                    cells = (((span.get("table_data") or {}).get("cells")) or [])
                    cell_text = " ".join((c.get("text") or "") for c in cells).lower()
                    if (
                        "trading symbol" in cell_text
                        or "trading symbol(s)" in cell_text
                        or "name of each exchange" in cell_text
                        or "exchange on which registered" in cell_text
                    ):
                        out.append(span)
            except Exception:
                continue
        return out
    except Exception:
        return []

