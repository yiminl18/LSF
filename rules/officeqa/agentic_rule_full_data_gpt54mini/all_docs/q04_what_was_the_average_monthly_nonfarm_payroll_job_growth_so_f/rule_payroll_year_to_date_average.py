def rule_payroll_year_to_date_average(doc: dict) -> list[dict]:
    try:
        import re

        def norm(text: str) -> str:
            return " ".join((text or "").split())

        def sentence_score(sentence: str, doc_year: str | None) -> int:
            text = norm(sentence)
            low = text.lower()
            if not text:
                return -1
            if "job growth" not in low and "job creation" not in low and "payroll" not in low:
                return -1
            if "average" not in low and "averag" not in low and "monthly gain" not in low:
                return -1

            score = 0
            if "nonfarm" in low:
                score += 2
            if "payroll job growth" in low or "payroll job creation" in low:
                score += 4
            if "job growth has averaged" in low or "job growth has average" in low:
                score += 5
            if "average monthly gain" in low:
                score += 4
            if "so far" in low or "thus far" in low or "year-to-date" in low or "ytd" in low:
                score += 6
            if "this year" in low or "this year's" in low or "current year" in low:
                score += 4
            if (
                "first quarter" in low
                or "second quarter" in low
                or "third quarter" in low
                or "fourth quarter" in low
            ):
                score += 2
            if "per month" in low:
                score += 3
            if re.search(r"\b\d{1,3}(?:,\d{3})\b", low):
                score += 2
            if doc_year and doc_year in low:
                score += 1
            return score

        doc_name = (doc.get("doc_name") or "").lower()
        year_match = re.search(r"treasury_bulletin_(\d{4})", doc_name)
        doc_year = year_match.group(1) if year_match else None
        sentence_split_re = re.compile(r"(?<=[.!?])\s+")

        paragraphs = doc.get("paragraphs", []) or []
        paragraph_candidates: list[tuple[int, int, dict]] = []
        for paragraph in paragraphs:
            raw_text = paragraph.get("text", "") or ""
            text = norm(raw_text)
            low = text.lower()
            if "payroll" not in low and "nonfarm" not in low:
                continue
            if "job growth" not in low and "job creation" not in low and "job growth has" not in low:
                continue
            if "average" not in low and "averag" not in low and "monthly gain" not in low:
                continue
            score = 0
            if "payroll job growth" in low or "payroll job creation" in low:
                score += 4
            if "nonfarm" in low:
                score += 2
            if "so far" in low or "thus far" in low or "year-to-date" in low or "ytd" in low:
                score += 6
            if "this year" in low or "this year's" in low or "current year" in low:
                score += 4
            if (
                "first quarter" in low
                or "second quarter" in low
                or "third quarter" in low
                or "fourth quarter" in low
            ):
                score += 2
            if "average monthly gain" in low:
                score += 4
            if "averaged" in low or "average" in low or "has average" in low:
                score += 2
            if "per month" in low:
                score += 3
            if re.search(r"\b\d{1,3}(?:,\d{3})\b", low):
                score += 2
            if doc_year and doc_year in low:
                score += 1
            paragraph_candidates.append((score, len(text), paragraph))

        if paragraph_candidates:
            paragraph_candidates.sort(key=lambda item: (-item[0], item[1]))
            best_paragraph = paragraph_candidates[0][2]
            paragraph_text = norm(best_paragraph.get("text", ""))
            sentences = sentence_split_re.split(paragraph_text)
            sentence_candidates: list[tuple[int, int, str]] = []
            for sentence in sentences:
                score = sentence_score(sentence, doc_year)
                if score < 0:
                    continue
                sentence_candidates.append((score, len(norm(sentence)), norm(sentence)))

            if sentence_candidates:
                sentence_candidates.sort(key=lambda item: (-item[0], item[1]))
                chosen = sentence_candidates[0][2]
                span = {"text": chosen}
                if best_paragraph.get("page_no") is not None:
                    span["page_no"] = best_paragraph.get("page_no")
                if best_paragraph.get("paragraph_no") is not None:
                    span["paragraph_no"] = best_paragraph.get("paragraph_no")
                return [span]

            return [best_paragraph]

        lines = doc.get("lines", []) or []
        for idx, line in enumerate(lines):
            text = norm(line.get("text", ""))
            low = text.lower()
            if "payroll" not in low and "nonfarm" not in low:
                continue
            if "job growth" not in low and "job creation" not in low and "monthly gain" not in low:
                continue
            if "average" not in low and "averag" not in low:
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
