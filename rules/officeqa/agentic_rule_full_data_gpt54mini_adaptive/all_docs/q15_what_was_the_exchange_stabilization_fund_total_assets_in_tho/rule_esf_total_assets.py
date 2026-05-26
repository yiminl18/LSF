import re


def rule_esf_total_assets(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        if not lines:
            return []

        def norm(text: str) -> str:
            return " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())

        def join_window(start: int, end: int) -> str:
            return " ".join(norm(lines[i].get("text", "")) for i in range(start, min(end, len(lines))))

        def is_esf_table_window(start: int) -> bool:
            window = join_window(start, start + 45)
            return (
                ("esf 1" in window or "table esf 1" in window)
                and (
                    "in thousands" in window
                    or "assets liabilities and capital" in window
                    or "presents the assets liabilities and capital" in window
                    or "balances as of" in window
                    or "balance sheet" in window
                )
            )

        start_candidates = []
        for idx in range(len(lines)):
            if is_esf_table_window(idx):
                start_candidates.append(idx)

        if not start_candidates and "exchange stabilization fund" in norm(doc.get("text", "")):
            # Fallback for documents where the table heading is split across more lines.
            start_candidates = [
                idx
                for idx in range(len(lines))
                if "exchange stabilization fund" in norm(lines[idx].get("text", ""))
            ]

        for start_idx in start_candidates:
            search_end = min(len(lines), start_idx + 260)
            total_idx = -1
            for idx in range(start_idx, search_end):
                window = join_window(idx, idx + 4)
                if "total assets" in window:
                    total_idx = idx
                    break

            if total_idx == -1:
                continue

            snippet_start = total_idx
            for idx in range(total_idx, min(len(lines), total_idx + 150)):
                if "total liabilities and capital" in norm(lines[idx].get("text", "")):
                    snippet_start = idx
                    break
            else:
                for idx in range(total_idx, min(len(lines), total_idx + 150)):
                    window = join_window(idx, idx + 4)
                    if "total liabilities and capital" in window:
                        snippet_start = idx
                        break

            end_idx = min(len(lines), snippet_start + 80)
            snippet_lines = lines[snippet_start:end_idx]
            snippet_text = "\n".join((line.get("text") or "").rstrip() for line in snippet_lines if line.get("text"))
            if snippet_text.strip():
                first_line = snippet_lines[0] if snippet_lines else {}
                last_line = snippet_lines[-1] if snippet_lines else {}
                span = {
                    "text": snippet_text,
                }
                if "page_no" in first_line:
                    span["page_no"] = first_line.get("page_no")
                if "line_no" in first_line:
                    span["line_no"] = first_line.get("line_no")
                if "page_no" in last_line:
                    span["end_page_no"] = last_line.get("page_no")
                if "line_no" in last_line:
                    span["end_line_no"] = last_line.get("line_no")
                return [span]

        return []
    except Exception:
        return []
