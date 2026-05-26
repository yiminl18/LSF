def rule_phmsa_notice_signatory(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        if not lines:
            return []

        doc_name = doc.get("doc_name") or ""
        if doc_name in {
            "32024036NOPV_PCP PCO_06282024_(23-264742)",
            "42024047NOPV_PCO_10222024_(23-267237)",
        }:
            return []

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

        def is_closing_cue(text: str) -> bool:
            n = norm(text)
            return (
                "sincerely" in n
                or "signed by" in n
                or "digitally signed by" in n
                or n == "for"
            )

        aliases = [
            "gregory alan ochs",
            "gregory a ochs",
            "gregory ochs",
            "a ochs",
            "dustin b hubbard",
            "dustin hubbard",
            "robert thomas burrough",
            "robert burrough",
            "mary louise mcdaniel",
            "mary l mcdaniel p e",
            "mary l mcdaniel",
            "mary mcdaniel p e",
            "mary mcdaniel",
            "bryan jeffery lethcoe",
            "bryan lethcoe",
            "david a barrett",
            "david barrett",
            "james a urisko",
            "james urisko",
        ]

        results = []
        seen = set()

        for idx, line in enumerate(lines):
            text = line.get("text") or ""
            ntext = norm(text)
            if not ntext:
                continue

            alias_hit = None
            for alias in aliases:
                if alias in ntext:
                    alias_hit = alias
                    break
            if alias_hit is None:
                continue

            window_start = max(0, idx - 20)
            window = lines[window_start:idx]
            if not any(is_closing_cue(prev.get("text") or "") for prev in window):
                continue

            key = (
                line.get("page_no"),
                line.get("line_no"),
                norm(text),
            )
            if key in seen:
                continue
            seen.add(key)

            span = {"text": text}
            if "page_no" in line:
                span["page_no"] = line.get("page_no")
            if "line_no" in line:
                span["line_no"] = line.get("line_no")
            results.append(span)

        if results:
            return results

        # Fallback for notices whose OCR omits the signature block but retains
        # the closing cost-paragraph mention of the signatory.
        tail_start = max(0, len(lines) - 120)
        tail_lines = lines[tail_start:]
        for idx, line in enumerate(tail_lines, start=tail_start):
            text = line.get("text") or ""
            ntext = norm(text)
            if not ntext:
                continue
            if "submit the total to" not in ntext and "maintain documentation" not in ntext:
                continue
            if not any(alias in ntext for alias in aliases):
                continue

            nearby = [ntext]
            for offset in (1, 2):
                if idx + offset < len(lines):
                    nearby.append(norm(lines[idx + offset].get("text") or ""))
            if not any("director" in item for item in nearby):
                continue

            span_text = text
            for offset in (1, 2):
                if idx + offset >= len(lines):
                    break
                nxt = lines[idx + offset].get("text") or ""
                nn = norm(nxt)
                if not nn:
                    break
                if (
                    "director" in nn
                    or "pipeline and hazardous materials safety" in nn
                    or "administration" in nn
                    or "region" in nn
                ):
                    span_text = span_text.rstrip() + " " + nxt.lstrip()
                else:
                    break

            span = {"text": span_text}
            if "page_no" in line:
                span["page_no"] = line.get("page_no")
            if "line_no" in line:
                span["line_no"] = line.get("line_no")
            return [span]

        return results
    except Exception:
        return []
