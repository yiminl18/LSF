def rule_page1_2_section12b_window_without_table(doc: dict) -> list[dict]:
    """Capture the short page 1-2 Section 12(b) span window when no exchange table is present."""
    try:
        import re

        texts = doc.get("texts", [])
        has_table = any(
            s.get("page_no", 999) <= 2
            and s.get("label") == "table"
            and "trading symbol" in s.get("text", "").lower()
            and "exchange" in s.get("text", "").lower()
            for s in texts
        )
        if has_table:
            return []

        stop_phrases = (
            "securities registered pursuant to section 12(g)",
            "indicate by check mark",
            "former name or former address",
            "emerging growth company",
            "item 1.",
            "item 2.",
            "item 5.",
            "item 8.",
            "item 9.",
        )

        out = []
        seen = set()
        for i, s in enumerate(texts):
            if (
                s.get("page_no", 999) <= 2
                and "securities registered pursuant to section 12(b)"
                in s.get("text", "").lower()
            ):
                page = s.get("page_no")
                page_items = [(k, sk) for k, sk in enumerate(texts) if sk.get("page_no") == page]
                note_titles = [
                    " ".join(sk.get("text", "").split())
                    for _, sk in page_items
                    if "notes due" in sk.get("text", "").lower()
                ]
                exchange_text = None
                for _, sk in page_items:
                    text = " ".join(sk.get("text", "").split())
                    low = text.lower()
                    if any(k in low for k in ("nasdaq", "new york stock exchange", "nyse", "exchange llc", "global select market")):
                        exchange_text = text
                        break

                symbol_text = None
                for idx, (k, sk) in enumerate(page_items):
                    text_low = sk.get("text", "").lower()
                    header_hit = "trading symbol" in text_low
                    if not header_hit and text_low.strip() == "trading" and idx + 1 < len(page_items):
                        next_low = page_items[idx + 1][1].get("text", "").lower()
                        header_hit = "symbol" in next_low
                    if not header_hit:
                        continue
                    for j in range(k + 1, min(len(texts), k + 8)):
                        sj = texts[j]
                        if sj.get("page_no") != page:
                            continue
                        candidate = " ".join(sj.get("text", "").split())
                        if re.fullmatch(r"[A-Z][A-Z0-9./%-]{0,7}", candidate):
                            symbol_text = candidate
                            break
                    if symbol_text:
                        break

                if len(note_titles) >= 3 and exchange_text and symbol_text:
                    structure = s.get("structure")
                    if not isinstance(structure, dict):
                        structure = {}
                        s["structure"] = structure
                    structure["level_index"] = i
                    parts = [f"{symbol_text} - {exchange_text}"]
                    parts.extend(f"{title} - {exchange_text}" for title in note_titles)
                    s["text"] = "Trading symbols and listing exchanges: " + "; ".join(parts)
                    idx = id(s)
                    if idx not in seen:
                        out.append(s)
                        seen.add(idx)
                    continue

                for j in range(i, min(len(texts), i + 48)):
                    sj = texts[j]
                    if sj.get("page_no") != page:
                        continue
                    structure = sj.get("structure")
                    if not isinstance(structure, dict):
                        structure = {}
                        sj["structure"] = structure
                    structure["level_index"] = j
                    idx = id(sj)
                    if idx not in seen:
                        out.append(sj)
                        seen.add(idx)
                    if j > i and any(p in sj.get("text", "").lower() for p in stop_phrases):
                        break
        return out
    except Exception:
        return []
