def rule_trading_symbol_section(doc: dict) -> list[dict]:
    try:
        import re

        lines = doc.get("lines") or []
        if not lines:
            return []

        def norm(text: str) -> str:
            return " ".join((text or "").split())

        header_re = re.compile(r"(?i)\bSecurities registered pursuant to Section\s*12\(b\)\s+of the Act\b")
        trading_re = re.compile(r"(?i)\bTrading\s+Symbol(?:\(s\))?\b")
        exchange_re = re.compile(
            r"(?i)\bName of (?:each\s+)?exchange on which registered\b|\bName of exchange on which registered\b"
        )
        terminator_re = re.compile(
            r"(?i)\bSecurities registered pursuant to Section\s*12\(g\)\b|"
            r"\bIndicate by check mark\b|"
            r"\bAs of\b|"
            r"\bThe registrant had\b|"
            r"\bDOCUMENTS INCORPORATED BY REFERENCE\b|"
            r"\bTABLE OF CONTENTS\b"
        )

        limit = min(len(lines), 180)
        start_idx = None
        for i in range(limit):
            window_items = []
            for j in range(i, min(limit, i + 3)):
                part = norm(lines[j].get("text"))
                if part:
                    window_items.append((j, part))
            if not window_items:
                continue
            window = " ".join(part for _, part in window_items)
            if header_re.search(window):
                for j, part in window_items:
                    if header_re.search(part):
                        start_idx = j
                        break
                if start_idx is None:
                    start_idx = i
                break
            if start_idx is None and (trading_re.search(window) or exchange_re.search(window)):
                title_j = None
                match_j = None
                for j, part in window_items:
                    if title_j is None and re.search(r"(?i)\btitle of each class\b", part):
                        title_j = j
                    if trading_re.search(part) or exchange_re.search(part):
                        match_j = j
                        break
                if title_j is not None:
                    start_idx = title_j
                elif match_j is not None:
                    start_idx = match_j
                else:
                    start_idx = i

        if start_idx is None:
            return []

        start_idx = max(0, start_idx)
        end_idx = min(len(lines), start_idx + 60)
        seen_answer_line = False
        for i in range(start_idx, end_idx):
            text = norm(lines[i].get("text"))
            if not text:
                continue
            low = text.lower()
            if (
                trading_re.search(text)
                or exchange_re.search(text)
                or "trading" in low
                or "symbol" in low
                or "exchange" in low
                or "title of each class" in low
            ):
                seen_answer_line = True
            if seen_answer_line and i > start_idx + 3 and terminator_re.search(text):
                end_idx = i
                break

        block = []
        seen = set()
        for i in range(start_idx, end_idx):
            text = norm(lines[i].get("text"))
            if not text:
                continue
            key = (text.lower(), lines[i].get("page_no"), lines[i].get("line_no"))
            if key in seen:
                continue
            seen.add(key)
            block.append(text)

        if not block:
            return []

        noise_re = re.compile(
            r"(?i)^(?:"
            r"title of each class|"
            r"trading|"
            r"symbol\(s\)|"
            r"symbol|"
            r"name of each exchange on which(?: registered)?|"
            r"name of exchange on which registered|"
            r"name of each exchange on which|"
            r"registered|"
            r"securities registered pursuant to section\s*12\(b\)(?: of the act)?|"
            r"securities registered pursuant to section\s*12\(g\)(?: of the act)?"
            r")$"
        )
        filtered = [line for line in block if not noise_re.match(line)]
        if filtered:
            block = filtered

        span = {"text": "\n".join(block)}
        if lines[start_idx].get("page_no") is not None:
            span["page_no"] = lines[start_idx]["page_no"]
        if lines[start_idx].get("line_no") is not None:
            span["line_no"] = lines[start_idx]["line_no"]
        return [span]
    except Exception:
        return []
