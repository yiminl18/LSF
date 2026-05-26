import re


def rule_long_term_debt_year_end(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        ordered = sorted(
            lines,
            key=lambda x: (
                x.get("page_no", 10**9),
                x.get("line_no", 10**9),
            ),
        )

        row_pat = re.compile(
            r"(?i)^\s*(?:total\s+)?long[- ]term debt"
            r"(?:"
            r"(?:\s+and\s+obligations\s+under\s+(?:capital|finance)\s+leases)"
            r"|(?:\s+obligations(?:\s+under\s+(?:capital|finance)\s+leases)?)"
            r"|(?:,\s*(?:net|net of current portion|less current portion|less:\s*current portion|excluding current portion))"
            r")?"
            r"(?:\s*\(\d+\))?\s*$"
        )
        bad_pat = re.compile(
            r"(?i)\b(?:including current portion|current portion of long[- ]term debt|"
            r"short[- ]term debt and current portion of long[- ]term debt|"
            r"due within one year|contractual obligations|future maturities|"
            r"consisted of the following|fair value|estimated fair value|"
            r"proceeds from long[- ]term debt|repayments of long[- ]term debt|"
            r"cash paid for interest|rights of holders|table of contents|"
            r"note\s+\d+\.?\s+long[- ]term debt)\b"
        )

        def clean(text: str) -> str:
            text = (text or "").replace("\u00a0", " ")
            return " ".join(text.split())

        def extract_amount(text: str):
            t = clean(text)
            if not t:
                return None
            if t in {"$", "€", "£"}:
                return None
            if re.fullmatch(r"[()\-–—]+", t):
                return "0"
            m = re.fullmatch(r"(?i)(?:[$€£]\s*)?(\d[\d,]*(?:\.\d+)?)", t)
            if m:
                return m.group(1)
            m = re.search(r"(?i)(\d[\d,]*(?:\.\d+)?)", t)
            if m and len(t) <= 20:
                return m.group(1)
            return None

        def add_span(text: str, source: dict) -> list[dict]:
            span = {"text": text}
            if source.get("page_no") is not None:
                span["page_no"] = source.get("page_no")
            if source.get("line_no") is not None:
                span["line_no"] = source.get("line_no")
            return [span]

        for i, item in enumerate(ordered):
            raw = clean(item.get("text"))
            if not raw:
                continue
            if not row_pat.match(raw):
                continue
            if bad_pat.search(raw):
                continue

            inline_amount = extract_amount(raw)
            if inline_amount is not None and inline_amount != raw:
                return add_span(inline_amount, item)

            currency_seen = False
            for j in range(i + 1, min(i + 5, len(ordered))):
                nxt = clean(ordered[j].get("text"))
                if not nxt:
                    continue
                if nxt in {"$", "€", "£"}:
                    currency_seen = True
                    continue
                amt = extract_amount(nxt)
                if amt is not None:
                    return add_span(amt, ordered[j])
                # Stop if the next nonblank line looks like prose rather than a number.
                if any(ch.isalpha() for ch in nxt):
                    break
                if currency_seen and re.fullmatch(r"[()\-–—]+", nxt):
                    return add_span("0", ordered[j])

        return [{"text": "0"}]
    except Exception:
        return []
