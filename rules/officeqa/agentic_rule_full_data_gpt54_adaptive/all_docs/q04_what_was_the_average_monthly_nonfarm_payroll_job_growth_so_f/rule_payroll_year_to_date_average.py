def rule_payroll_year_to_date_average(doc: dict) -> list[dict]:
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
                or "jobsonnonfarmpayrolls" in flat
                or "payrolljobs" in flat
                or "payrollemployment" in flat
                or "jobgrowth" in flat
                or "jobcreation" in flat
                or "jobgains" in flat
                or ("jobs" in flat and ("added" in flat or "created" in flat or "gains" in flat))
            )
            if not context_hits:
                return -1

            average_hits = (
                "average" in flat
                or "averaged" in flat
                or "averaging" in flat
                or "permonth" in flat
                or "monthlyaverage" in flat
                or "monthlyincrease" in flat
                or "monthlygain" in flat
            )
            if not average_hits:
                return -1

            explicit_time_hits = (
                "thisyear" in flat
                or "thisyears" in flat
                or "sofarin" in flat
                or "currentyear" in flat
                or "yearsofar" in flat
                or "thusfar" in flat
                or "yeartodate" in flat
                or "throughthefirst" in flat
                or bool(re.search(r"through(?:january|february|march|april|may|june|july|august|september|october|november|december)", flat))
                or bool(re.search(r"endingin(?:january|february|march|april|may|june|july|august|september|october|november|december)", flat))
                or bool(re.search(r"first(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)months", flat))
                or "monthsoftheyear" in flat
            )
            quarter_hits = (
                "firstquarter" in flat
                or "secondquarter" in flat
                or "thirdquarter" in flat
                or "fourthquarter" in flat
                or bool(re.search(r"first(?:\d+|one|two|three|four)quarters", flat))
            )
            current_year_hits = explicit_time_hits or (doc_year and f"of{doc_year}" in flat)
            if not current_year_hits and not (quarter_hits and doc_year and doc_year in flat):
                return -1

            value = 0
            if "nonfarmpayroll" in flat or "jobsonnonfarmpayrolls" in flat:
                value += 4
            if "payrolljobs" in flat or "payrollemployment" in flat:
                value += 3
            if "jobgrowthhasaveraged" in flat or "paceofjobcreationaveraged" in flat:
                value += 4
            if "averageof" in flat or "averaged" in flat or "permonth" in flat:
                value += 3
            if (
                "thisyear" in flat
                or "yearsofar" in flat
                or "yeartodate" in flat
                or "sofarin" in flat
                or "thusfar" in flat
                or re.search(r"through(?:january|february|march|april|may|june|july|august|september|october|november|december)", flat)
                or re.search(r"endingin(?:january|february|march|april|may|june|july|august|september|october|november|december)", flat)
            ):
                value += 4
            if "firstquarter" in flat or "first" in flat and "months" in flat:
                value += 3
            if doc_year and doc_year in flat:
                value += 1
            return value

        doc_name = (doc.get("doc_name") or "").lower()
        year_match = re.search(r"treasury_bulletin_(\d{4})", doc_name)
        doc_year = year_match.group(1) if year_match else None
        best: tuple[int, int, dict] | None = None

        for para in doc.get("paragraphs") or []:
            para_text = normalize(para.get("text") or "")
            if not para_text:
                continue
            candidates = []
            para_sents = sentences(para_text)
            for idx, sent in enumerate(para_sents):
                sent = normalize(sent)
                base_score = score(sent, doc_year)
                if base_score >= 0:
                    candidates.append((base_score, len(sent), sent))
                    continue
                options = []
                if idx > 0:
                    options.append(normalize(para_sents[idx - 1] + " " + sent))
                if idx + 1 < len(para_sents):
                    options.append(normalize(sent + " " + para_sents[idx + 1]))
                for option in options:
                    sent_score = score(option, doc_year)
                    if sent_score >= 0:
                        candidates.append((sent_score, len(option), option))
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
            base = normalize(line.get("text") or "")
            if not base:
                continue
            window_parts = []
            for j in range(max(0, idx - 2), min(len(lines), idx + 4)):
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
