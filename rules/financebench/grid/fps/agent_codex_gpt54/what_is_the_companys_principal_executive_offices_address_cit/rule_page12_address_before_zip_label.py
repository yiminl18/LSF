def rule_page12_address_before_zip_label(doc: dict) -> list[dict]:
    """Match the early-page address span immediately before a standalone zip-code label."""
    try:
        import re

        street_re = re.compile(
            r"\b(?:street|st\.?|drive|dr\.?|avenue|ave\.?|road|rd\.?|way|plaza|"
            r"boulevard|blvd\.?|lane|ln\.?|suite|ste\.?|park|tower)\b",
            re.I,
        )

        results = []
        texts = doc.get("texts", [])
        valid_labels = {
            "(zip code)",
            "zip code",
            "(address of principal executive offices) (zip code)",
            "(address of principal executive offices and zip code)",
        }

        for i, span in enumerate(texts):
            if span.get("page_no", 999) > 2:
                continue

            label_text = " ".join((span.get("text") or "").split())
            if label_text.lower() not in valid_labels:
                continue
            if i - 1 < 0:
                continue

            prev = texts[i - 1]
            if prev.get("page_no") != span.get("page_no"):
                continue

            prev_text = " ".join((prev.get("text") or "").split())
            prev_lowered = prev_text.lower()
            if any(
                marker in prev_lowered
                for marker in (
                    "employer identification",
                    "i.r.s.",
                    "irs employer",
                    "commission file",
                    "telephone number",
                    "former name",
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
