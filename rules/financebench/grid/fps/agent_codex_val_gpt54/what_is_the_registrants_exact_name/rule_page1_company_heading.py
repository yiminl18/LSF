def rule_page1_company_heading(doc: dict) -> list[dict]:
    """Match early page-1 heading-style spans whose text looks like a company name."""
    try:
        import re

        texts = doc.get("texts", [])
        out = []
        bad_phrases = (
            "united states",
            "securities and exchange commission",
            "form 10-",
            "form 8-k",
            "form 10 q",
            "form 10 k",
            "current report",
            "annual report",
            "quarterly report",
            "transition report",
            "commission file",
            "date of report",
            "exact name of registrant",
            "state or other jurisdiction",
            "registrant's telephone",
            "registrant’s telephone",
            "address of principal",
            "title of each class",
            "trading symbol",
            "name of each exchange",
            "washington, d.c.",
            "news release",
            "summary financial results",
            "financial outlook",
            "table of contents",
            "signatures",
            "item ",
            "part ",
        )

        for span in texts[:18]:
            if (span.get("page_no") or 999) != 1:
                continue
            if span.get("label") not in {"text", "section_header"}:
                continue

            text = (span.get("text") or "").replace("\n", " ").strip()
            low = text.lower()
            if not text or len(text) > 120 or len(text.split()) > 8:
                continue
            if any(phrase in low for phrase in bad_phrases):
                continue
            if re.fullmatch(r"[\W\d]+", text or ""):
                continue
            if sum(ch.isalpha() for ch in text) < 4:
                continue

            size = span.get("size") or 0
            level = ((span.get("structure") or {}).get("level")) or ""
            if size >= 12 or level in {"H1", "H2", "H3"}:
                out.append(span)
        return out
    except Exception:
        return []
