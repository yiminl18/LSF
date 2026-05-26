def rule_payroll_early_year_gain(doc: dict) -> list[dict]:
    try:
        import re

        def normalize(text: str) -> str:
            text = text or ""
            text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        def dense(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", (text or "").lower())

        def has_number(text: str) -> bool:
            return bool(re.search(r"(?<!\w)-?\d[\d,]*", text or ""))

        def sentences(text: str) -> list[str]:
            text = normalize(text)
            if not text:
                return []
            parts = re.split(r"(?<=[.!?])\s+", text)
            return [part.strip() for part in parts if part.strip()]

        def score(text: str, doc_year: str | None) -> int:
            text = normalize(text)
            flat = dense(text)
            if not text or not has_number(text):
                return -1

            context_hits = (
                "nonfarmpayroll" in flat
                or "payrolljobs" in flat
                or "payrollemployment" in flat
                or "jobgrowth" in flat
                or "jobcreation" in flat
            )
            if not context_hits:
                return -1

            month_hits = (
                "january" in flat
                or "decemberandjanuary" in flat
            )
            if not month_hits:
                return -1

            action_hits = (
                "added" in flat
                or "increased" in flat
                or "rose" in flat
                or "grew" in flat
                or "soared" in flat
                or "fell" in flat
                or "drop" in flat
                or "decline" in flat
            )
            if not action_hits:
                return -1

            value = 0
            if "january" in flat or "injanuary" in flat:
                value += 3
            if "decemberandjanuary" in flat:
                value += 2
            if "payrolljobs" in flat or "payrollemployment" in flat or "nonfarmpayroll" in flat:
                value += 4
            if "averagejobgrowth" in flat or "jobincreasesaveraged" in flat or "payrolljobspermonth" in flat:
                value += 2
            if doc_year and doc_year in flat:
                value += 1
            return value

        doc_name = (doc.get("doc_name") or "").lower()
        year_match = re.search(r"treasury_bulletin_(\d{4})", doc_name)
        doc_year = year_match.group(1) if year_match else None
        best: tuple[int, int, dict] | None = None

        for para in doc.get("paragraphs") or []:
            text = normalize(para.get("text") or "")
            if not text:
                continue
            sents = sentences(text)
            if not sents:
                continue
            candidates = []
            for idx, sent in enumerate(sents):
                sent = normalize(sent)
                if not sent:
                    continue
                options = [sent]
                if idx > 0:
                    options.append(normalize(sents[idx - 1] + " " + sent))
                if idx + 1 < len(sents):
                    options.append(normalize(sent + " " + sents[idx + 1]))
                for option in options:
                    option_score = score(option, doc_year)
                    if option_score >= 0:
                        candidates.append((option_score, len(option), option))
            if not candidates:
                continue
            candidates.sort(key=lambda item: (-item[0], item[1]))
            chosen = candidates[0][2]
            span = {"text": chosen}
            if para.get("page_no") is not None:
                span["page_no"] = para["page_no"]
            if para.get("paragraph_no") is not None:
                span["paragraph_no"] = para["paragraph_no"]
            candidate = (candidates[0][0], len(chosen), span)
            if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                best = candidate

        if best is not None:
            return [best[2]]

        lines = doc.get("lines") or []
        for idx, line in enumerate(lines):
            window_parts = []
            for j in range(max(0, idx - 2), min(len(lines), idx + 5)):
                part = normalize(lines[j].get("text") or "")
                if part:
                    window_parts.append(part)
            window = normalize(" ".join(window_parts))
            window_score = score(window, doc_year)
            if window_score < 0:
                continue
            span = {"text": window}
            if line.get("page_no") is not None:
                span["page_no"] = line["page_no"]
            if line.get("line_no") is not None:
                span["line_no"] = line["line_no"]
            candidate = (window_score, len(window), span)
            if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                best = candidate

        return [best[2]] if best is not None else []
    except Exception:
        return []
