import re


def rule_ffo4_total_receipts_table(doc: dict) -> list[dict]:
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

        def extract_numbers(text: str) -> list[str]:
            return re.findall(r"\d[\d,]*", text or "")

        def has_fytd_narrative() -> bool:
            for item in lines:
                low = norm(item.get("text") or "")
                if "federal receipts" not in low and "total federal receipts" not in low:
                    continue
                if "fytd" not in low and "fiscal year to date" not in low and "fiscal year-to-date" not in low:
                    continue
                if any(key in low for key in ("were $", "totaled $", "jumped by $", "were up by $", "up by $")):
                    return True
            return False

        if has_fytd_narrative():
            return []

        for item in lines:
            low = norm(item.get("text") or "")
            if re.search(r"\bfiscal year\s+202[4-9]\s+to date\b", low):
                return []

        heading_idx = None
        for idx, line in enumerate(lines):
            low = norm(line.get("text") or "")
            if "table ffo-4" in low and "summary of u.s. government receipts by source and outlays by agency" in low:
                heading_idx = idx
                break
        if heading_idx is None:
            for idx, line in enumerate(lines):
                low = norm(line.get("text") or "")
                if "table ffo-4" in low and "summary of u.s. government receipts by source and outlays by" in low:
                    heading_idx = idx
                    break
        if heading_idx is None:
            return []

        results = []
        search_limit = min(len(lines), heading_idx + 800)
        for idx in range(heading_idx, search_limit):
            low = norm(lines[idx].get("text") or "")
            if "total receipts" not in low:
                continue
            nums = []
            for j in range(idx, min(len(lines), idx + 8)):
                nums.extend(extract_numbers(lines[j].get("text") or ""))
                if len(nums) >= 4:
                    break
            if len(nums) < 4:
                continue
            answer = nums[3]
            span = {"text": answer}
            if lines[idx].get("page_no") is not None:
                span["page_no"] = lines[idx].get("page_no")
            if lines[idx].get("line_no") is not None:
                span["line_no"] = lines[idx].get("line_no")
            results.append(span)
            break

        return results
    except Exception:
        return []
