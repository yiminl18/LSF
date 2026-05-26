import re


def rule_principal_executive_offices_narrative(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        if not lines:
            return []

        def norm(value: str) -> str:
            return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()

        phrase_re = re.compile(
            r"\b(principal executive offices|principal address is)\b",
            re.IGNORECASE,
        )
        start_re = re.compile(
            r"\b(?:our|the)\s+principal executive offices\b|\bprincipal executive offices\b|\bprincipal address is\b",
            re.IGNORECASE,
        )
        located_re = re.compile(
            r"\b(located at|are located at|principal address is|are at)\b",
            re.IGNORECASE,
        )
        street_re = re.compile(
            r"\b(street|st\.|road|rd\.|avenue|ave\.|drive|dr\.|boulevard|blvd\.|plaza|lane|ln\.|"
            r"way|highway|hwy\.|court|ct\.|parkway|pkwy\.|circle|cir\.|suite|ste\.|floor|building|"
            r"bldg\.|tower|center|centre)\b",
            re.IGNORECASE,
        )
        country_re = re.compile(
            r"\b(united states|united kingdom|canada|switzerland|japan|mexico|ireland|australia)\b",
            re.IGNORECASE,
        )
        address_cue_re = re.compile(
            r"(\d{1,5}\s+\w+|\b\d{5}(?:-\d{4})?\b|[A-Z]{2}\s+\d{5}|"
            r"[A-Z][a-z]+,\s*[A-Z][A-Za-z. ]+|[A-Z][A-Za-z.'-]+\s+[A-Z]{2})"
        )
        stop_re = re.compile(
            r"\b(located in the u\.s\.|located in the us|principal executive offices are located in the u\.s\.|"
            r"principal executive offices are located in the us|principal executive offices are in the u\.s\.|"
            r"principal executive offices are in the us)\b",
            re.IGNORECASE,
        )

        for i, line in enumerate(lines):
            if line.get("page_no") not in (1, 2):
                continue

            current = norm(line.get("text", ""))
            if not current:
                continue

            low_current = current.lower()
            if not phrase_re.search(low_current) and not located_re.search(low_current):
                continue

            window_sources = []
            for j in range(i, min(len(lines), i + 2)):
                raw = norm(lines[j].get("text", ""))
                if raw:
                    window_sources.append(lines[j])
            if not window_sources:
                continue

            window_text = " ".join(norm(src.get("text", "")) for src in window_sources)
            low = window_text.lower()
            if not phrase_re.search(low) and not located_re.search(low):
                continue
            if stop_re.search(low):
                continue
            if not (street_re.search(window_text) or country_re.search(window_text) or address_cue_re.search(window_text)):
                continue

            start_match = start_re.search(window_text)
            if start_match:
                window_text = window_text[start_match.start() :].strip()

            span = {"text": window_text}
            if window_sources[0].get("page_no") is not None:
                span["page_no"] = window_sources[0]["page_no"]
            if window_sources[0].get("line_no") is not None:
                span["line_no"] = window_sources[0]["line_no"]
            return [span]

        return []
    except Exception:
        return []
