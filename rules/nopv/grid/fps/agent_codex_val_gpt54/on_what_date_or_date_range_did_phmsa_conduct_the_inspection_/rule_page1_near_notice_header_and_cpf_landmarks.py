def rule_page1_near_notice_header_and_cpf_landmarks(doc: dict) -> list[dict]:
    """Match a top-of-page PHMSA inspection paragraph that sits below the notice header and Dear or CPF landmarks."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:20]):
            text = " ".join((span.get("text") or "").split())
            lowered = text.lower()
            if not (
                span.get("page_no") == 1
                and span.get("label") == "text"
                and "phmsa" in lowered
                and ("inspect" in lowered or "inspection" in lowered)
                and "investigation" not in lowered
            ):
                continue

            prev_notice = [s for s in texts[max(0, i - 10):i] if s.get("page_no") == 1]
            prev_landmarks = [s for s in texts[max(0, i - 6):i] if s.get("page_no") == 1]
            if any("notice of probable violation" in ((s.get("text") or "").lower()) for s in prev_notice) and any(
                ("dear " in ((s.get("text") or "").lower()) or "cpf" in ((s.get("text") or "").lower()))
                for s in prev_landmarks
            ):
                out.append(span)
        return out
    except Exception:
        return []
