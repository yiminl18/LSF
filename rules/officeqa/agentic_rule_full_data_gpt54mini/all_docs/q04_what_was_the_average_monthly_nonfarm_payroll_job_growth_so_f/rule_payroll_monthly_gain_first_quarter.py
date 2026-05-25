def rule_payroll_monthly_gain_first_quarter(doc: dict) -> list[dict]:
    try:
        import re

        def norm(text: str) -> str:
            return " ".join((text or "").split())

        doc_name = (doc.get("doc_name") or "").lower()
        year_match = re.search(r"treasury_bulletin_(\d{4})", doc_name)
        doc_year = year_match.group(1) if year_match else None
        sentence_split_re = re.compile(r"(?<=[.!?])\s+")

        def sentence_score(sentence: str) -> int:
            text = norm(sentence)
            low = text.lower()
            if not text:
                return -1
            if "payroll" not in low and "nonfarm" not in low and "job growth" not in low:
                return -1
            if "average" not in low and "averag" not in low and "monthly gain" not in low:
                return -1

            score = 0
            if "average monthly gain" in low:
                score += 6
            if "per month" in low:
                score += 4
            if "this year" in low or "this year's" in low or "current year" in low:
                score += 5
            if "so far" in low or "thus far" in low or "year-to-date" in low:
                score += 5
            if "first quarter" in low:
                score += 4
            if "second quarter" in low or "third quarter" in low or "fourth quarter" in low:
                score += 1
            if "payroll job growth" in low or "payroll job creation" in low:
                score += 3
            if "nonfarm" in low:
                score += 1
            if re.search(r"\b\d{1,3}(?:,\d{3})\b", low):
                score += 2
            if doc_year and doc_year in low:
                score += 1
            return score

        paragraphs = doc.get("paragraphs", []) or []
        candidates: list[tuple[int, int, dict, str]] = []
        for paragraph in paragraphs:
            raw_text = paragraph.get("text", "") or ""
            text = norm(raw_text)
            low = text.lower()
            if "payroll" not in low and "nonfarm" not in low and "job growth" not in low:
                continue
            if "average" not in low and "averag" not in low and "monthly gain" not in low:
                continue
            if "per month" not in low and "monthly gain" not in low and "average" not in low:
                continue
            score = 0
            if "average monthly gain" in low:
                score += 6
            if "per month" in low:
                score += 4
            if "this year" in low or "this year's" in low or "current year" in low:
                score += 5
            if "so far" in low or "thus far" in low or "year-to-date" in low:
                score += 5
            if "first quarter" in low:
                score += 4
            if "payroll job growth" in low or "payroll job creation" in low:
                score += 3
            if "nonfarm" in low:
                score += 1
            if re.search(r"\b\d{1,3}(?:,\d{3})\b", low):
                score += 2
            if doc_year and doc_year in low:
                score += 1
            for sentence in sentence_split_re.split(text):
                sent_score = sentence_score(sentence)
                if sent_score < 0:
                    continue
                candidates.append((score + sent_score, len(norm(sentence)), paragraph, norm(sentence)))

        if candidates:
            candidates.sort(key=lambda item: (-item[0], item[1]))
            best = candidates[0]
            span = {"text": best[3]}
            if best[2].get("page_no") is not None:
                span["page_no"] = best[2].get("page_no")
            if best[2].get("paragraph_no") is not None:
                span["paragraph_no"] = best[2].get("paragraph_no")
            return [span]

        lines = doc.get("lines", []) or []
        for idx, line in enumerate(lines):
            text = norm(line.get("text", ""))
            low = text.lower()
            if "payroll" not in low and "nonfarm" not in low and "job growth" not in low:
                continue
            if "average monthly gain" not in low and "per month" not in low and "averaged" not in low:
                continue
            start = max(0, idx - 1)
            end = min(len(lines), idx + 3)
            combined = " ".join(
                norm(lines[j].get("text", ""))
                for j in range(start, end)
                if norm(lines[j].get("text", ""))
            )
            if combined:
                span = {"text": combined}
                if line.get("page_no") is not None:
                    span["page_no"] = line.get("page_no")
                if line.get("line_no") is not None:
                    span["line_no"] = line.get("line_no")
                return [span]

        return []
    except Exception:
        return []
