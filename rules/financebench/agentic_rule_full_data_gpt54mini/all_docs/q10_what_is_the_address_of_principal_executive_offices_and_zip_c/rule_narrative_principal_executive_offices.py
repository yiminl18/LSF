import re


def rule_narrative_principal_executive_offices(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        if not lines:
            return []

        normalized = []
        for idx, line in enumerate(lines):
            text = line.get("text") or ""
            match_text = (
                text.replace("\u2019", "'")
                .replace("\u2018", "'")
                .replace("\u00a0", " ")
                .replace("\u202f", " ")
                .replace("\u2009", " ")
            )
            match_text = re.sub(r"\s+", " ", match_text).strip()
            normalized.append(
                {
                    "idx": idx,
                    "page_no": line.get("page_no"),
                    "line_no": line.get("line_no"),
                    "text": text,
                    "lower": match_text.lower(),
                }
            )

        block_keywords = (
            "telephone",
            "employer identification",
            "commission file",
            "exact name of registrant",
            "securities registered",
            "trading symbol",
            "indicate by check mark",
            "table of contents",
        )

        spans = []
        seen = set()

        for i, item in enumerate(normalized):
            low = item["lower"]
            if not low:
                continue
            if "principal executive offices" not in low:
                continue
            if "located at" not in low and "located in" not in low:
                continue
            if not re.search(r"\d", low):
                continue

            collected = []
            start = i
            end = min(len(normalized), i + 3)
            for j in range(start, end):
                cur = normalized[j]
                text = cur["text"].strip()
                low2 = cur["lower"]
                if not text:
                    continue
                if any(pat in low2 for pat in block_keywords) and j != i:
                    continue
                if ("principal executive offices" in low2 and ("located at" in low2 or "located in" in low2)) or j == i:
                    collected.append(cur)
                    continue

                # Pull in continuation lines that look like address fragments, zip codes, or telephone-only noise
                # when they are part of the same sentence block.
                if re.search(r"\d", text) or "," in text or re.fullmatch(r"\d{5}(?:-\d{4})?", text):
                    collected.append(cur)

            if not collected:
                continue

            if not any(re.search(r"\d", x["text"]) for x in collected):
                continue

            key = tuple((x["page_no"], x["line_no"], x["text"]) for x in collected)
            if key in seen:
                continue
            seen.add(key)

            joined = " ".join(x["text"].strip() for x in collected if x["text"].strip())
            first = collected[0]
            last = collected[-1]
            spans.append(
                {
                    "text": joined,
                    "page_no": first["page_no"],
                    "line_no": first["line_no"],
                    "end_line_no": last["line_no"],
                }
            )

        return spans
    except Exception:
        return []
