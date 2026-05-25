def rule_reuters_michigan_consumer_sentiment_latest_reading(doc: dict) -> list[dict]:
    try:
        import re

        def norm(text: str) -> str:
            return " ".join((text or "").split())

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
        topic_re = re.compile(
            r"\b(?:Reuters/Michigan consumer sentiment index|"
            r"Michigan Index of Consumer Sentiment|consumer sentiment index)\b",
            re.I,
        )
        value_re = re.compile(r"\b\d+(?:\.\d+)?\b")
        verb_re = re.compile(
            r"\b(?:stood at|fell to|declined to|rose to|reached|was|trended higher)\b",
            re.I,
        )
        sentence_split_re = re.compile(r"(?<=[.!?])\s+")

        doc_name = (doc.get("doc_name") or "").lower()
        doc_year_match = re.search(r"treasury_bulletin_(\d{4})_(\d{2})", doc_name)
        doc_year = int(doc_year_match.group(1)) if doc_year_match else None
        doc_month = int(doc_year_match.group(2)) if doc_year_match else None
        if doc_year is not None and (doc_year > 2024 or (doc_year == 2024 and doc_month and doc_month >= 12)):
            return []

        spans = []
        seen = set()

        def add_span(text: str, meta: dict | None = None) -> None:
            cleaned = norm(text)
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

        def sentence_score(sentence: str) -> tuple[int, int] | None:
            text = norm(sentence)
            low = text.lower()
            if not text or not value_re.search(low) or not verb_re.search(low):
                return None

            score = 0
            if "stood at" in low:
                score += 6
            if "fell to" in low or "declined to" in low:
                score += 5
            if "rose to" in low or "reached" in low:
                score += 3
            if "trended higher" in low:
                score += 1
            if month_re.search(low):
                score += 3
            if "early" in low or "late" in low or "mid" in low:
                score += 1

            latest_date = 0
            for match in month_year_extract_re.finditer(text):
                month_key = match.group("month").lower()
                year = int(match.group("year"))
                latest_date = max(latest_date, year * 12 + month_to_num.get(month_key, 0))
            if latest_date == 0 and doc_year is not None and month_re.search(text):
                for month_match in month_re.finditer(text):
                    month_key = month_match.group(0).lower()
                    latest_date = max(
                        latest_date, doc_year * 12 + month_to_num.get(month_key, 0)
                    )
            if latest_date == 0:
                return None

            return latest_date * 100 + score, len(text)

        paragraph_candidates: list[tuple[int, int, str, dict]] = []
        for paragraph in doc.get("paragraphs") or []:
            raw_text = paragraph.get("text") or ""
            text = norm(raw_text)
            low = text.lower()
            if not text or not topic_re.search(low):
                continue
            if "consumer sentiment" not in low:
                continue

            best: tuple[int, int, str] | None = None
            for sentence in sentence_split_re.split(text):
                scored = sentence_score(sentence)
                if scored is None:
                    continue
                score, length = scored
                candidate = (score, length, norm(sentence))
                if best is None or candidate[0] > best[0] or (
                    candidate[0] == best[0] and candidate[1] < best[1]
                ):
                    best = candidate

            if best is not None:
                paragraph_candidates.append((best[0], best[1], best[2], paragraph))

        if paragraph_candidates:
            paragraph_candidates.sort(key=lambda item: (-item[0], item[1]))
            best = paragraph_candidates[0]
            add_span(best[2], best[3])
            return spans

        lines = doc.get("lines") or []
        for idx, line in enumerate(lines):
            raw_text = line.get("text") or ""
            text = norm(raw_text)
            low = text.lower()
            if "consumer sentiment" not in low:
                continue
            if not topic_re.search(low):
                continue
            start = max(0, idx - 1)
            end = min(len(lines), idx + 3)
            combined = " ".join(
                norm(lines[j].get("text") or "")
                for j in range(start, end)
                if norm(lines[j].get("text") or "")
            )
            if combined:
                add_span(combined, line)
                return spans

        return []
    except Exception:
        return []
