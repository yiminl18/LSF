import re


def rule_ffo1_total_surplus_or_deficit_table(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        if not lines:
            return []

        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").strip()).lower()

        def clean_text(text: str) -> str:
            return (text or "").replace("−", "-").replace("--", "-")

        def extract_signed_numbers(text: str) -> list[str]:
            return re.findall(r"-?\d[\d,]*", clean_text(text))

        def make_span_from_line(text: str, line: dict) -> dict:
            span = {"text": text}
            if line.get("page_no") is not None:
                span["page_no"] = line.get("page_no")
            if line.get("line_no") is not None:
                span["line_no"] = line.get("line_no")
            return span

        for idx, line in enumerate(lines):
            low = norm(lines[idx].get("text") or "")
            if "total surplus or deficit" not in low:
                continue
            if "on-budget" in low or "off-budget" in low:
                continue

            collected: list[tuple[int, str]] = []
            for j in range(idx, min(len(lines), idx + 5)):
                raw = lines[j].get("text") or ""
                nums = extract_signed_numbers(raw)
                if not nums:
                    continue
                for num in nums:
                    collected.append((j, num))
                if len(collected) >= 2:
                    break
            if not collected:
                continue

            answer_idx, answer = collected[-1]
            answer_line = lines[answer_idx]
            return [make_span_from_line(answer, answer_line)]

        return []
    except Exception:
        return []
