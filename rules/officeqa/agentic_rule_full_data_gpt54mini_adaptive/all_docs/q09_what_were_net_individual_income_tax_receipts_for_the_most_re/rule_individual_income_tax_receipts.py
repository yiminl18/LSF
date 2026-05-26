import re


def rule_individual_income_tax_receipts(doc: dict) -> list[dict]:
    try:
        results = []
        seen = set()

        patterns = [
            re.compile(r"individual income tax receipts(?:, net of refunds)?(?:,)? were \$[\d,]+(?:\.\d+)?", re.I),
            re.compile(r"individual income taxes.*?receipts, net of refunds, were \$[\d,]+(?:\.\d+)?", re.I),
            re.compile(r"individual income taxes.*?individual income tax receipts.*?were \$[\d,]+(?:\.\d+)?", re.I),
        ]

        def add_span(text, extra=None):
            if not text:
                return
            norm_text = " ".join(str(text).split())
            key = (norm_text, tuple(sorted(extra.items())) if extra else ())
            if key in seen:
                return
            seen.add(key)
            span = {"text": norm_text}
            if extra:
                span.update(extra)
            results.append(span)

        for paragraph in doc.get("paragraphs", []) or []:
            text = paragraph.get("text", "") or ""
            if "individual income tax" not in text.lower():
                continue
            if not any(p.search(text.replace("\n", " ")) for p in patterns):
                continue
            add_span(
                text,
                {
                    "page_no": paragraph.get("page_no"),
                    "paragraph_no": paragraph.get("paragraph_no"),
                },
            )

        if results:
            return results

        for line in doc.get("lines", []) or []:
            text = line.get("text", "") or ""
            if "individual income tax" not in text.lower():
                continue
            if not any(p.search(text.replace("\n", " ")) for p in patterns):
                continue
            add_span(
                text,
                {
                    "page_no": line.get("page_no"),
                    "line_no": line.get("line_no"),
                },
            )

        if results:
            return results

        text = doc.get("text", "") or ""
        low = text.lower()
        if "individual income tax" in low and "$" in text:
            for pat in patterns:
                m = pat.search(text.replace("\n", " "))
                if m:
                    add_span(m.group(0))
                    break
            if results:
                return results

        table_anchors = [
            "net budget receipts, by source",
            "table ffo-2",
            "on-budget and off-budget receipts by source",
            "onbudget and off-budget receipts by source",
            "budget receipts by source",
            "receipts by source",
            "budget receipts by",
        ]
        if any(anchor in low for anchor in table_anchors) and (
            "individual" in low or "income taxes" in low or "withheld" in low or "refunds" in low
        ):
            for anchor in table_anchors:
                idx = low.find(anchor)
                if idx == -1:
                    continue
                start = max(0, idx - 250)
                end = min(len(text), idx + 3500)
                window = " ".join(text[start:end].split())
                if window:
                    add_span(window)
                    break

        return results
    except Exception:
        return []
