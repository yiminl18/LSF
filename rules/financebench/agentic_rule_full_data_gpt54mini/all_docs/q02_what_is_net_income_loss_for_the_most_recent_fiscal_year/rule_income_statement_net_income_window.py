import re


def rule_income_statement_net_income_window(doc: dict) -> list[dict]:
    try:
        lines = [ln for ln in (doc.get("lines") or []) if isinstance(ln, dict)]
        if not lines:
            return []

        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()

        ordered = sorted(
            lines,
            key=lambda ln: (
                int(ln.get("page_no") or 0),
                int(ln.get("line_no") or 0),
            ),
        )

        label_re = re.compile(
            r"\b(?:"
            r"consolidated\s+net\s+(?:income|earnings)(?:\s*\(loss\))?(?:\s+attributable\s+to\b(?!\s*noncontrolling\b).*)?|"
            r"net\s+(?:income|earnings)(?:\s*\(loss\))?(?!\s+per\s+share)(?:\s+attributable\s+to\b(?!\s*noncontrolling\b).*)?|"
            r"net\s+loss(?:\s+attributable\s+to\b(?!\s*noncontrolling\b).*)?"
            r")\b",
            re.IGNORECASE,
        )
        numeric_re = re.compile(r"^[\s$\-(),.\d\u2013\u2014]+$")
        year_re = re.compile(r"\b(?:19|20)\d{2}\b|\byear ended\b", re.IGNORECASE)

        def is_numericish(text: str) -> bool:
            low = norm(text)
            return bool(low) and (bool(numeric_re.fullmatch(low)) or bool(year_re.search(low)))

        def is_context(text: str) -> bool:
            low = norm(text)
            if not low:
                return False
            if is_numericish(low):
                return True
            if re.match(
                r"^(?:less:|total|net income|net earnings|net loss|"
                r"consolidated net income|adjustments to reconcile|"
                r"basic net income|diluted net income|shares used to compute|"
                r"year ended|years ended|fiscal year|trailing twelve months|"
                r"calculation of return on assets|return on assets|"
                r"calculation of return on investment|return on investment|"
                r"selected financial data|other financial measures|"
                r"net income per share|earnings per share)",
                low,
                re.IGNORECASE,
            ):
                return True
            if len(low.split()) <= 4 and re.fullmatch(r"[\d,().$\-\s]+", low):
                return True
            return False

        candidates = []
        seen = set()

        for i, ln in enumerate(ordered):
            text = norm(ln.get("text"))
            if not text:
                continue

            low = text.lower()
            if "table of contents" in low:
                continue

            joined = " ".join(
                norm(ordered[j].get("text"))
                for j in range(i, min(len(ordered), i + 3))
                if norm(ordered[j].get("text"))
            )
            if not joined or not (label_re.search(text) or label_re.search(joined)):
                continue

            has_numeric_neighbor = any(
                is_numericish(norm(ordered[j].get("text")))
                for j in range(i + 1, min(len(ordered), i + 5))
            )
            if not has_numeric_neighbor:
                continue

            context_start = max(0, i - 20)
            context_end = min(len(ordered), i + 20)
            context_text = " ".join(
                norm(ordered[j].get("text"))
                for j in range(context_start, context_end)
                if norm(ordered[j].get("text"))
            )
            context_low = context_text.lower()

            score = 0
            if re.search(r"\btrailing twelve months\b", context_low):
                score += 120
            if re.search(r"\b(?:for the )?(?:year|years) ended\b|\bfiscal year\b", context_low):
                score += 100
            if re.search(
                r"\b(?:consolidated|condensed)?\s*statements?\s+of\s+"
                r"(?:income(?:\s*\(loss\))?|operations(?:\s*\(loss\))?|earnings(?:\s*\(loss\))?|comprehensive\s+income(?:\s*\(loss\))?)",
                context_low,
            ) or re.search(
                r"\b(?:consolidated|condensed)?\s*statement\s+of\s+"
                r"(?:income(?:\s*\(loss\))?|operations(?:\s*\(loss\))?|earnings(?:\s*\(loss\))?|comprehensive\s+income(?:\s*\(loss\))?)",
                context_low,
            ):
                score += 85
            if re.search(r"\bcalculation of return on assets\b|\breturn on assets\b", context_low):
                score += 95
            if re.search(r"\bcalculation of return on investment\b|\breturn on investment\b", context_low):
                score += 90
            if re.search(r"\bnet income per share\b|\bearnings per share\b", context_low):
                score += 65
            if re.search(r"\bselected financial data\b|\bother financial measures\b", context_low):
                score += 40
            if re.search(
                r"\bnet income from continuing operations\b|\bnet loss from discontinued operations\b",
                context_low,
            ):
                score += 130
            if re.search(r"\badjusted net income\b", context_low):
                continue
            if re.search(r"\bthree months ended\b|\bsix months ended\b|\bquarterly period ended\b", context_low):
                score -= 80
            if re.search(r"\bcash flows from operating activities\b|\badjustments to reconcile net income\b", context_low):
                score -= 35
            if re.search(r"\btable of contents\b", context_low):
                score -= 80

            if re.search(r"\bwas\b|\bwere\b", low):
                score -= 20

            if re.fullmatch(r"net income(?:\s*\(loss\))?", low):
                score += 55
            elif re.fullmatch(r"net earnings(?:\s*\(loss\))?", low):
                score += 50
            elif low.startswith("consolidated net income"):
                score += 48
            elif low.startswith("net income"):
                score += 40
            elif low.startswith("net earnings"):
                score += 38
            elif low.startswith("net loss"):
                score += 32

            word_count = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
            if word_count > 10 and re.search(r"\bwas\b|\bwere\b", low):
                continue
            if word_count > 12 and not low.startswith(
                (
                    "net income",
                    "net earnings",
                    "consolidated net income",
                    "net loss",
                )
            ):
                continue
            if word_count <= 5 and " was " not in f" {low} " and " were " not in f" {low} ":
                score += 20
            elif word_count > 12:
                score -= 12

            if "noncontrolling" in context_low:
                score -= 35
            if "attributable" in low and "noncontrolling" not in low:
                score -= 100
            if "attributable" in low and "noncontrolling" in low:
                score -= 20
            if re.search(r"\b\d{4}\b", context_low):
                score += 12

            if score <= 0 and not re.search(r"\byear ended\b|\bfiscal year\b|\btrailing twelve months\b", context_low):
                continue

            start = max(0, i - 2)
            end = min(len(ordered), i + 6)
            snippet_lines = []
            for j in range(start, end):
                t = norm(ordered[j].get("text"))
                if not t:
                    continue
                if j == i or is_context(t) or label_re.search(t):
                    snippet_lines.append(t)

            snippet = "\n".join(snippet_lines).strip()
            if not snippet or not any(ch.isdigit() for ch in snippet):
                continue

            key = snippet.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append((score, i, snippet))

        if not candidates:
            return []

        candidates.sort(key=lambda item: (-item[0], item[1], len(item[2])))
        out = []
        seen_out = set()
        for _, i, snippet in candidates:
            if snippet in seen_out:
                continue
            seen_out.add(snippet)
            ln = ordered[i]
            out.append(
                {
                    "page_no": ln.get("page_no"),
                    "line_no": ln.get("line_no"),
                    "text": snippet,
                }
            )
            if len(out) >= 3:
                break

        return out
    except Exception:
        return []
