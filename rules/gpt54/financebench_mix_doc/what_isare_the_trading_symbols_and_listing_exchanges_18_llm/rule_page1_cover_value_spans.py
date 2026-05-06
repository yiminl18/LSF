def rule_page1_cover_value_spans(doc: dict) -> list[dict]:
    """Retrieve page-1 cover-page body/table spans containing trading symbol and exchange values."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            try:
                if span.get("page_no") != 1:
                    continue
                label = span.get("label")
                txt = (span.get("text") or "")
                path = ((span.get("structure") or {}).get("path_text") or "")
                low = txt.lower()
                path_low = path.lower()
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
                    continue
                if label != "text":
                    continue
                if not path:
                    continue
                if len(txt.split()) > 12:
                    continue
                if (
                    "securities registered pursuant to section 12(b)" in path_low
                    or "trading symbol" in path_low
                    or "name of each exchange" in path_low
                    or "exchange on which registered" in path_low
                    or any(k in low for k in [
                        "nasdaq", "new york stock exchange", "chicago stock exchange",
                        "global select market", "nyse"
                    ])
                    or re.fullmatch(r"[A-Z]{1,5}(?:\d{2}|/\d{2})?", txt.strip()) is not None
                ):
                    out.append(span)
            except Exception:
                continue
        return out
    except Exception:
        return []

