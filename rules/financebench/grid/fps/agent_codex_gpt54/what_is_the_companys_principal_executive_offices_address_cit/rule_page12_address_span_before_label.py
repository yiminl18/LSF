def rule_page12_address_span_before_label(doc: dict) -> list[dict]:
    """Match the early-page address span immediately before a standalone principal-offices label."""
    try:
        import re

        street_re = re.compile(
            r"\b(?:street|st\.?|drive|dr\.?|avenue|ave\.?|road|rd\.?|way|plaza|"
            r"boulevard|blvd\.?|lane|ln\.?|suite|ste\.?|park|tower)\b",
            re.I,
        )

        results = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("page_no", 999) > 2:
                continue

            label_text = " ".join((span.get("text") or "").split())
            lowered = label_text.lower()
            if "address of principal executive offices" not in lowered:
                continue
            if len(lowered) > 90 or street_re.search(label_text) or "," in label_text:
                continue

            if i - 1 < 0:
                continue
            prev = texts[i - 1]
            if prev.get("page_no") != span.get("page_no"):
                continue

            prev_text = " ".join((prev.get("text") or "").split())
            prev_lowered = prev_text.lower()
            if not prev_text:
                continue
            if any(
                marker in prev_lowered
                for marker in (
                    "jurisdiction",
                    "commission file",
                    "exact name of registrant",
                    "employer identification",
                    "i.r.s.",
                    "irs employer",
                    "zip code",
                    "telephone number",
                    "former name",
                    "trading symbol",
                    "exchange on which",
                    "securities registered",
                    "pursuant to section",
                )
            ):
                continue
            if not (
                street_re.search(prev_text)
                or "," in prev_text
                or re.fullmatch(r"[A-Z]{2}", prev_text.strip())
            ):
                continue

            results.append(prev)

        return results
    except Exception:
        return []
