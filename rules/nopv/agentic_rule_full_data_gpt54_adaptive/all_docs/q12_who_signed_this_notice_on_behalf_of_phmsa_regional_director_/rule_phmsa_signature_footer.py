def rule_phmsa_signature_footer(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        if not lines:
            return []

        def clean(text: str) -> str:
            return " ".join((text or "").replace("\x0c", " ").split())

        def norm(text: str) -> str:
            text = (text or "").lower()
            out = []
            last_space = False
            for ch in text:
                if ch.isalnum():
                    out.append(ch)
                    last_space = False
                else:
                    if not last_space:
                        out.append(" ")
                        last_space = True
            return " ".join("".join(out).split())

        def nonempty_indices(start: int, stop: int, step: int) -> list[int]:
            found = []
            idx = start
            while (idx < stop if step > 0 else idx > stop):
                if clean(lines[idx].get("text") or ""):
                    found.append(idx)
                idx += step
            return found

        bad_title_phrases = (
            "region director may extend",
            "submit to the director",
            "director of the southern region",
            "received this notice from a different regional director",
            "director of the western region within",
            "director of the western region copies",
            "region director for approval",
            "central region director for review",
            "to gregory ochs director central region",
        )

        for idx, line in enumerate(lines):
            text = clean(line.get("text") or "")
            ntext = norm(text)
            if not ntext or "director" not in ntext:
                continue
            if not (
                "region" in ntext
                or "office of pipeline safety" in ntext
                or " ops" in f" {ntext} "
                or ntext.startswith("acting director")
            ):
                continue
            if any(phrase in ntext for phrase in bad_title_phrases):
                continue

            lookahead = []
            for j in nonempty_indices(idx, min(len(lines), idx + 8), 1):
                lookahead.append((j, clean(lines[j].get("text") or ""), norm(lines[j].get("text") or "")))
            if not lookahead:
                continue

            phmsa_line_idx = None
            followup_cue = False
            for j, raw, nraw in lookahead:
                if (
                    "pipeline and hazardous materials safety administration" in nraw
                    or "hazardous materials safety administration" in nraw
                    or nraw == "phmsa"
                ):
                    phmsa_line_idx = j
                if (
                    "enclosures" in nraw
                    or "response options" in nraw
                    or nraw == "cc"
                    or nraw.startswith("cc ")
                ):
                    followup_cue = True

            if phmsa_line_idx is None:
                continue

            lookback = []
            for j in nonempty_indices(idx - 1, max(-1, idx - 12), -1):
                raw = clean(lines[j].get("text") or "")
                lookback.append((j, raw, norm(raw)))

            closing_cue_idx = None
            for j, raw, nraw in lookback:
                if (
                    "sincerely" in nraw
                    or "signed by" in nraw
                    or "digitally signed by" in nraw
                    or nraw.startswith("for ")
                ):
                    closing_cue_idx = j
                    break

            if closing_cue_idx is None and not followup_cue:
                continue

            for j, raw, nraw in lookback:
                if not nraw.startswith("for "):
                    continue
                if len(raw.split()) < 2:
                    continue
                span = {
                    "text": f'The full signatory phrase, including the word "For", is: {raw}.'
                }
                if "page_no" in lines[j]:
                    span["page_no"] = lines[j].get("page_no")
                if "line_no" in lines[j]:
                    span["line_no"] = lines[j].get("line_no")
                return [span]

            start_idx = idx
            if closing_cue_idx is not None:
                start_idx = closing_cue_idx
            else:
                # Keep a tight block when OCR drops the "Sincerely" cue but leaves the
                # name/title/footer cluster intact.
                for j, raw, nraw in reversed(lookback):
                    if (
                        any(ch.isdigit() for ch in raw)
                        or "page " in nraw
                        or "cpf " in nraw
                        or "in your correspondence" in nraw
                    ):
                        break
                    start_idx = j

            snippet_lines = []
            for j in range(start_idx, phmsa_line_idx + 1):
                raw = clean(lines[j].get("text") or "")
                if raw:
                    snippet_lines.append(raw)

            snippet = "\n".join(snippet_lines).strip()
            if not snippet:
                continue

            span = {"text": snippet}
            if "page_no" in line:
                span["page_no"] = line.get("page_no")
            if "line_no" in line:
                span["line_no"] = line.get("line_no")
            return [span]

        return []
    except Exception:
        return []
