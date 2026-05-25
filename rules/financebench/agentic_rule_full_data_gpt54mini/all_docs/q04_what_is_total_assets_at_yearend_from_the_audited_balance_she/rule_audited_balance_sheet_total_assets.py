import re


def rule_audited_balance_sheet_total_assets(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        if not lines:
            return []

        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").strip()).lower()

        def join_window(start: int, end: int) -> str:
            parts = []
            for i in range(max(0, start), min(len(lines), end)):
                text = (lines[i].get("text") or "").strip()
                if text:
                    parts.append(text)
            return "\n".join(parts).strip()

        def extract_year_end_amount(total_assets_idx: int) -> str:
            pieces = []
            for i in range(total_assets_idx, min(len(lines), total_assets_idx + 6)):
                text = (lines[i].get("text") or "").strip()
                if not text:
                    continue
                if i == total_assets_idx:
                    parts = re.split(r"(?i)total assets", text, maxsplit=1)
                    if len(parts) > 1:
                        text = parts[1]
                    else:
                        text = ""
                pieces.append(text)
            blob = " ".join(pieces)
            nums = re.findall(r"\(?\d[\d,]*(?:\.\d+)?\)?", blob)
            for num in nums:
                cleaned = num.strip("()")
                if cleaned:
                    return cleaned
            return ""

        heading_terms = (
            "consolidated balance sheets",
            "balance sheets",
            "balance sheet",
            "consolidated statements of financial position",
            "statement of financial position",
        )
        context_terms = (
            "cash and cash equivalents",
            "accounts receivable",
            "receivables",
            "inventory",
            "property and equipment",
            "goodwill",
            "other assets",
            "current assets",
            "non-current assets",
            "liabilities",
            "equity",
            "stockholders",
            "shareholders",
        )

        audit_present = False
        for line in lines:
            low = norm(line.get("text") or "")
            if "we have audited" not in low:
                continue
            if "balance sheet" in low or "balance sheets" in low or "financial position" in low:
                audit_present = True
                break
        if not audit_present:
            return []

        spans = []
        seen = set()

        for idx, line in enumerate(lines):
            text = (line.get("text") or "").strip()
            low = norm(text)
            if not text:
                continue
            if "unaudited" in low:
                continue
            if "we have audited" in low:
                continue
            if not any(term in low for term in heading_terms):
                continue

            search_end = min(len(lines), idx + 180)
            for j in range(idx, search_end):
                cur_text = (lines[j].get("text") or "").strip()
                cur_low = norm(cur_text)
                if "total assets" not in cur_low:
                    continue
                if "does not exceed" in cur_low or "percent of total assets" in cur_low:
                    continue

                context_start = max(idx, j - 12)
                context_end = min(len(lines), j + 12)
                context_text = " ".join(
                    norm(lines[k].get("text") or "") for k in range(context_start, context_end)
                )
                if not any(term in context_text for term in context_terms):
                    continue

                amount = extract_year_end_amount(j)
                window_text = amount if amount else join_window(j - 3, j + 6)
                if not window_text:
                    continue

                key = (line.get("page_no"), line.get("line_no"), window_text)
                if key in seen:
                    continue
                seen.add(key)

                span = {"text": window_text}
                if lines[j].get("page_no") is not None:
                    span["page_no"] = lines[j].get("page_no")
                if lines[j].get("line_no") is not None:
                    span["line_no"] = lines[j].get("line_no")
                spans.append(span)
                break

            if spans:
                break

        return spans
    except Exception:
        return []
