def rule_recent_real_gdp_growth(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return " ".join((text or "").split()).lower()

        paragraphs = doc.get("paragraphs", []) or []
        candidates = []
        for paragraph in paragraphs:
            text = norm(paragraph.get("text", ""))
            if "real gdp growth" not in text:
                continue
            if "annual rate" not in text and "annualized" not in text:
                continue
            score = 0
            if text.startswith("economic growth"):
                score += 3
            if "advance estimate" in text:
                score += 2
            if "percent" in text:
                score += 2
            if "in the " in text and "quarter" in text:
                score += 1
            candidates.append((score, len(text), paragraph))

        if candidates:
            candidates.sort(key=lambda item: (-item[0], item[1]))
            return [candidates[0][2]]

        lines = doc.get("lines", []) or []
        for idx, line in enumerate(lines):
            text = norm(line.get("text", ""))
            if "real gdp growth" not in text:
                continue
            if "annual rate" not in text and "annualized" not in text:
                continue
            end = min(len(lines), idx + 4)
            combined = " ".join(
                (lines[j].get("text", "") or "").strip()
                for j in range(idx, end)
                if (lines[j].get("text", "") or "").strip()
            ).strip()
            if combined:
                span = {"text": combined}
                if "page_no" in line:
                    span["page_no"] = line.get("page_no")
                if "line_no" in line:
                    span["line_no"] = line.get("line_no")
                return [span]

        return []
    except Exception:
        return []
