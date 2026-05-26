def rule_payroll_chart_current_period(doc: dict) -> list[dict]:
    try:
        import re

        def normalize(text: str) -> str:
            text = text or ""
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        def dense(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", (text or "").lower())

        doc_name = (doc.get("doc_name") or "").lower()
        year_match = re.search(r"treasury_bulletin_(\d{4})", doc_name)
        doc_year = year_match.group(1) if year_match else ""
        year_short = doc_year[-2:] if doc_year else ""

        # Prefer explicit textual evidence when the document states a year-to-date
        # or first-month average directly; reserve the chart rule for chart-heavy docs.
        for para in doc.get("paragraphs") or []:
            text = normalize(para.get("text") or "")
            flat = dense(text)
            if not text:
                continue
            has_context = (
                "nonfarmpayroll" in flat
                or "payrollemployment" in flat
                or "payrolljobs" in flat
                or "jobgrowth" in flat
                or "jobgains" in flat
                or "jobcreation" in flat
            )
            has_average = "average" in flat or "averaged" in flat or "permonth" in flat
            has_ytd = (
                "sofar" in flat
                or "thisyear" in flat
                or "yearsofar" in flat
                or "yeartodate" in flat
                or "throughthefirst" in flat
                or bool(re.search(r"through(?:january|february|march|april|may|june|july|august|september|october|november|december)", flat))
                or bool(re.search(r"endingin(?:january|february|march|april|may|june|july|august|september|october|november|december)", flat))
                or bool(re.search(r"first(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)months", flat))
                or (doc_year and f"sofarin{doc_year}" in flat)
            )
            has_first_month = (
                ("january" in flat or "february" in flat)
                and has_context
                and has_average
            )
            if has_context and has_average and (has_ytd or has_first_month):
                return []

        lines = doc.get("lines") or []
        best: tuple[int, int, dict] | None = None

        for idx, line in enumerate(lines):
            base = normalize(line.get("text") or "")
            flat = dense(base)
            if "payrollemployment" not in flat and "establishmentemployment" not in flat:
                continue

            window_raw = []
            saw_average_axis = False
            for j in range(idx, min(len(lines), idx + 60)):
                raw = normalize(lines[j].get("text") or "")
                if not raw:
                    continue
                raw_flat = dense(raw)
                if j > idx + 4 and ("unemploymentrate" in raw_flat or "nonfarmproductivity" in raw_flat):
                    break
                if (
                    "averagemonthlychangeinthousands" in raw_flat
                    or "averagemonthlychangeinthousandsfromendofquartertoendofquarter" in raw_flat
                    or "monthlyaverageforyearshownandmonthlyamountsinthousands" in raw_flat
                ):
                    saw_average_axis = True
                window_raw.append(raw)

            if not saw_average_axis or len(window_raw) < 6:
                continue

            window_text = "\n".join(window_raw)
            if doc_year and doc_year not in window_text:
                if not year_short:
                    continue
                if not re.search(rf"(?<!\d){re.escape(year_short)}(?:\s*-\s*[IVX]+|\s*\*)?", window_text):
                    continue

            value = 0
            if "payrollemployment" in flat:
                value += 3
            if "establishmentemployment" in flat:
                value += 2
            if doc_year and doc_year in window_text:
                value += 2
            if year_short and re.search(rf"(?<!\d){re.escape(year_short)}(?:\s*-\s*[IVX]+|\s*\*)?", window_text):
                value += 2

            span = {"text": window_text}
            if line.get("page_no") is not None:
                span["page_no"] = line["page_no"]
            if line.get("line_no") is not None:
                span["line_no"] = line["line_no"]

            candidate = (value, len(window_text), span)
            if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                best = candidate

        return [best[2]] if best is not None else []
    except Exception:
        return []
