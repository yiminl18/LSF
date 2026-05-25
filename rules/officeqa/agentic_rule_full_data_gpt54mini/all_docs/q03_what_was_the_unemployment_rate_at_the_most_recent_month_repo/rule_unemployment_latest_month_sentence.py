def rule_unemployment_latest_month_sentence(doc: dict) -> list[dict]:
    try:
        import re

        months = (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        )
        month_to_num = {month.lower(): idx + 1 for idx, month in enumerate(months)}
        month_re = re.compile(r"\b(?:%s)\b" % "|".join(months), re.I)
        month_year_re = re.compile(r"\b(?:%s)\s+\d{4}\b" % "|".join(months), re.I)
        month_year_extract_re = re.compile(
            r"\b(?P<month>%s)\s+(?P<year>\d{4})\b" % "|".join(months), re.I
        )
        percent_re = re.compile(r"\b\d+(?:\.\d+)?\s*percent\b", re.I)
        unemployment_re = re.compile(
            r"\b(?:headline\s+|civilian\s+)?unemployment rate\b", re.I
        )
        verb_re = re.compile(
            r"\b(?:stood at|declined to|fell to|edged up to|ticked up to|rose to|"
            r"dipped to|remained at|held at|was)\b",
            re.I,
        )
        sentence_split_re = re.compile(r"(?<=[.!?])\s+")
        doc_name = (doc.get("doc_name") or "").lower()
        doc_year_match = re.search(r"treasury_bulletin_(\d{4})_(\d{2})", doc_name)
        doc_year = int(doc_year_match.group(1)) if doc_year_match else None

        spans = []
        seen = set()

        def add_span(text: str, meta: dict | None = None) -> None:
            cleaned = " ".join((text or "").split())
            if not cleaned:
                return
            key = cleaned.lower()
            if key in seen:
                return
            seen.add(key)
            span = {"text": cleaned}
            if meta is not None:
                if meta.get("page_no") is not None:
                    span["page_no"] = meta.get("page_no")
                if meta.get("paragraph_no") is not None:
                    span["paragraph_no"] = meta.get("paragraph_no")
                if meta.get("line_no") is not None:
                    span["line_no"] = meta.get("line_no")
            spans.append(span)

        def score_sentence(sentence: str) -> int:
            text = " ".join((sentence or "").split())
            low = text.lower()
            if "unemployment rate" not in low:
                return -1
            if not month_re.search(low):
                return -1
            if not percent_re.search(low):
                return -1
            if (
                "of those unemployed" in low
                or "share of the unemployed" in low
                or "long-term" in low
                or "27 weeks" in low
                or "broader measure" in low
            ):
                return -1

            directness = 0
            if unemployment_re.search(text):
                directness += 4
            if verb_re.search(text):
                directness += 4
            if low.startswith("the unemployment rate") or low.startswith(
                "unemployment rate"
            ):
                directness += 2
            if month_year_re.search(text):
                directness += 1
            if "headline unemployment rate" in low or "civilian unemployment rate" in low:
                directness += 1
            if "u-6" in low or "underemployment" in low:
                directness -= 5

            latest_date = 0
            first_rate_pos = low.find("unemployment rate")
            for match in month_year_extract_re.finditer(text):
                if first_rate_pos >= 0 and match.start() < first_rate_pos:
                    prefix = text[: match.start()].strip().lower()
                    if not re.search(
                        r"\b(?:in|by|as of|during|from|after|since|at|on|for)\b",
                        prefix,
                    ):
                        # Likely a page heading or page marker rather than the
                        # sentence introducing the unemployment rate.
                        continue
                month_key = match.group("month").lower()
                year = int(match.group("year"))
                latest_date = max(
                    latest_date, year * 12 + month_to_num.get(month_key, 0)
                )
            if latest_date == 0 and doc_year is not None:
                for month_match in month_re.finditer(text):
                    month_key = month_match.group(0).lower()
                    latest_date = max(
                        latest_date, doc_year * 12 + month_to_num.get(month_key, 0)
                    )

            return latest_date * 100 + directness

        paragraphs = doc.get("paragraphs") or []
        paragraph_candidates = []
        for paragraph in paragraphs:
            raw_text = paragraph.get("text") or ""
            text = " ".join(raw_text.split())
            if "unemployment rate" not in text.lower():
                continue

            sentences = sentence_split_re.split(text)
            best = None
            for sentence in sentences:
                score = score_sentence(sentence)
                if score < 0:
                    continue
                candidate = (score, len(sentence), sentence)
                if best is None or candidate[0] > best[0] or (
                    candidate[0] == best[0] and candidate[1] < best[1]
                ):
                    best = candidate
            if best is not None:
                paragraph_candidates.append((best[0], best[1], best[2], paragraph))
            elif month_re.search(text) and percent_re.search(text):
                paragraph_candidates.append((0, len(text), text, paragraph))

        if paragraph_candidates:
            paragraph_candidates.sort(key=lambda item: (-item[0], item[1]))
            add_span(paragraph_candidates[0][2], paragraph_candidates[0][3])
            return spans

        lines = doc.get("lines") or []
        for idx, line in enumerate(lines):
            raw_text = line.get("text") or ""
            text = " ".join(raw_text.split())
            low = text.lower()
            if "unemployment rate" not in low:
                continue
            if not month_re.search(low) or not percent_re.search(low):
                continue
            start = max(0, idx - 1)
            end = min(len(lines), idx + 3)
            combined = " ".join(
                " ".join(((lines[j].get("text") or "").split()))
                for j in range(start, end)
                if (lines[j].get("text") or "").strip()
            )
            if combined:
                add_span(combined, line)

        return spans
    except Exception:
        return []
