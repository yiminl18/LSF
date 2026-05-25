import re


def rule_income_statement_revenue_window(doc: dict) -> list[dict]:
    try:
        lines = [ln for ln in (doc.get("lines") or []) if isinstance(ln, dict)]
        if not lines:
            return []

        ordered = sorted(
            lines,
            key=lambda ln: (
                int(ln.get("page_no") or 0),
                int(ln.get("line_no") or 0),
            ),
        )

        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()

        def alpha_words(text: str) -> int:
            return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text or ""))

        def is_numericish(text: str) -> bool:
            low = norm(text)
            return bool(low) and bool(re.fullmatch(r"[\s$\-(),.\d\u2013\u2014]+", low))

        def has_annual_context(text: str) -> bool:
            low = norm(text).lower()
            if not low:
                return False
            return bool(
                re.search(
                    r"\b(?:year|years|fiscal year|fiscal years|52 weeks|53 weeks)\s+ended\b"
                    r"|\b52\s+weeks\s+ended\b|\b53\s+weeks\s+ended\b",
                    low,
                )
            )

        def has_quarter_context(text: str) -> bool:
            low = norm(text).lower()
            return bool(
                re.search(
                    r"\b(?:three|six|nine|twelve)\s+months\s+ended\b|\bquarter\s+ended\b",
                    low,
                )
            )

        year_any_re = re.compile(r"\b(?:19|20)\d{2}\b")

        heading_re = re.compile(
            r"\b(?:condensed\s+)?consolidated\s+statements?\s+of\s+(?:income|operations|earnings)"
            r"(?:\s*\(unaudited\))?\b|\bstatements?\s+of\s+(?:income|operations|earnings)\b"
            r"|\bstatement\s+of\s+operations\s+data\b",
            re.IGNORECASE,
        )
        annual_hint_re = re.compile(
            r"\b(?:year|years|fiscal year|fiscal years|52 weeks|53 weeks)\s+ended\b",
            re.IGNORECASE,
        )
        unit_re = re.compile(r"\b(?:million|millions|billion|billions|thousand|thousands)\b", re.IGNORECASE)
        year_token_re = re.compile(r"^(?:19|20)\d{2}$")
        revenue_total_re = re.compile(
            r"^\s*(?:"
            r"total\s+net\s+sales|"
            r"total\s+revenues?|"
            r"total\s+revenue|"
            r"net\s+revenues?|"
            r"net\s+sales|"
            r"revenue:|"
            r"revenue|"
            r"product\s+sales|"
            r"sales"
            r")\s*$",
            re.IGNORECASE,
        )
        revenue_label_re = re.compile(
            r"\b(?:total\s+net\s+sales|total\s+revenues?|total\s+revenue|net\s+revenues?|net\s+sales|revenue|product\s+sales|sales)\b",
            re.IGNORECASE,
        )
        stop_re = re.compile(
            r"\b(?:cost of revenue|cost of sales|gross profit|operating expenses|"
            r"selling, general and administrative|research and development|"
            r"operating income|operating loss|other income|income before income taxes|"
            r"income tax|net income|net loss)\b",
            re.IGNORECASE,
        )

        def candidate_score(label: str, window_text: str) -> int:
            low = norm(label).lower()
            ctx = window_text.lower()
            score = 0

            if re.search(r"\btotal\s+net\s+sales\b", low):
                score += 140
            if re.search(r"\btotal\s+revenue(s)?\b", low):
                score += 138
            if re.search(r"\btotal\s+revenues?\b", low):
                score += 136
            if re.search(r"\bnet\s+revenues?\b", low):
                score += 128
            if re.search(r"\bnet\s+sales\b", low):
                score += 126
            if re.search(r"\brevenue\s*:\s*$", low) or re.fullmatch(r"revenue", low):
                score += 94
            elif re.search(r"\brevenue\b", low):
                score += 90
            if re.search(r"\bproduct\s+sales\b", low):
                score += 84
            if re.search(r"\bsales\b", low):
                score += 70

            if re.search(r"\btable of contents\b", ctx):
                score -= 120
            if has_quarter_context(ctx) and not annual_hint_re.search(ctx):
                score -= 120
            if annual_hint_re.search(ctx):
                score += 30
            if re.search(r"\baudited\b|\bfinancial statements\b", ctx):
                score += 20
            if re.search(r"\bconsolidated statements?\s+of\b", ctx):
                score += 28
            if re.search(r"\bstatement of operations data\b", ctx):
                score += 18
            if alpha_words(low) > 8:
                score -= 45
            elif alpha_words(low) <= 3:
                score += 12
            if re.search(r"\b(?:revenue|sales|net sales|net revenue)\b", low) and not is_numericish(label):
                score += 8

            return score

        best = None
        seen = set()

        for i, ln in enumerate(ordered):
            text = norm(ln.get("text"))
            if not text:
                continue
            low = text.lower()
            if "table of contents" in low:
                continue
            if not heading_re.search(text):
                continue

            context_start = max(0, i - 4)
            context_end = min(len(ordered), i + 42)
            context_text = " ".join(
                norm(ordered[j].get("text"))
                for j in range(context_start, context_end)
                if norm(ordered[j].get("text"))
            )
            year_hits = year_any_re.findall(context_text)
            annual_context = bool(annual_hint_re.search(context_text) or len(set(year_hits)) >= 2)
            if not annual_context:
                continue
            if has_quarter_context(context_text) and not annual_hint_re.search(context_text):
                continue

            for j in range(i + 1, min(len(ordered), i + 46)):
                cand = norm(ordered[j].get("text"))
                if not cand:
                    continue
                cand_low = cand.lower()
                if "table of contents" in cand_low:
                    continue
                if is_numericish(cand) and not revenue_label_re.search(cand_low):
                    continue
                if stop_re.search(cand_low) and not revenue_label_re.search(cand_low):
                    break

                joined = cand
                if j + 1 < len(ordered):
                    joined = f"{cand} {norm(ordered[j + 1].get('text'))}".strip()
                if not revenue_label_re.search(joined):
                    continue
                if is_numericish(cand) and not re.search(r"\btotal\s+net\s+sales\b|\btotal\s+revenue(s)?\b|\bnet\s+revenues?\b|\bnet\s+sales\b", cand_low):
                    continue

                numeric_window = False
                for k in range(j, min(len(ordered), j + 6)):
                    nxt = norm(ordered[k].get("text"))
                    if not nxt:
                        continue
                    if is_numericish(nxt):
                        numeric_window = True
                        break
                if not numeric_window:
                    continue

                window_start = max(0, i - 8)
                window_end = min(len(ordered), j + 5)
                unit_match = unit_re.search(context_text) or unit_re.search(joined)
                unit_word = ""
                if unit_match:
                    unit_word = unit_match.group(0).lower().rstrip("s")
                has_currency = any(
                    norm(ordered[k].get("text")) == "$"
                    or norm(ordered[k].get("text")).startswith("$")
                    for k in range(window_start, window_end)
                )
                snippet_lines = []
                for k in range(window_start, window_end):
                    line_text = norm(ordered[k].get("text"))
                    if not line_text:
                        continue
                    line_low = line_text.lower()
                    if "table of contents" in line_low:
                        continue
                    keep = False
                    if k == i or k == j:
                        keep = True
                    elif heading_re.search(line_text):
                        keep = True
                    elif annual_hint_re.search(line_text):
                        keep = True
                    elif unit_re.search(line_text):
                        keep = True
                    elif k < j and year_token_re.fullmatch(line_text):
                        keep = True
                    elif k > j and is_numericish(line_text):
                        keep = True
                    if keep:
                        if k > j and is_numericish(line_text) and not year_token_re.fullmatch(line_text):
                            if has_currency and not line_text.startswith("$"):
                                line_text = f"$ {line_text}"
                            if unit_word:
                                line_text = f"{line_text} {unit_word}"
                        snippet_lines.append(line_text)

                snippet = "\n".join(snippet_lines).strip()
                if not snippet or not any(ch.isdigit() for ch in snippet):
                    continue

                key = (int(ln.get("page_no") or 0), int(ln.get("line_no") or 0), snippet.lower())
                if key in seen:
                    continue
                seen.add(key)

                score = candidate_score(cand, context_text + " " + snippet)
                if best is None or score > best[0] or (score == best[0] and len(snippet) < len(best[3])):
                    best = (
                        score,
                        int(ordered[i].get("page_no") or 0),
                        int(ordered[j].get("line_no") or 0),
                        snippet,
                    )

        if best is None:
            return []

        return [
            {
                "page_no": best[1],
                "line_no": best[2],
                "text": best[3],
            }
        ]
    except Exception:
        return []
