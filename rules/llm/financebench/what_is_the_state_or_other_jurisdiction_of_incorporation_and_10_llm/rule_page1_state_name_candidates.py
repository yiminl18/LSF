def rule_page1_state_name_candidates(doc: dict) -> list[dict]:
    """Match short page-1 spans that look like a state/jurisdiction value."""
    try:
        import re
        states = {
            "alabama","alaska","arizona","arkansas","california","colorado","connecticut","delaware",
            "florida","georgia","hawaii","idaho","illinois","indiana","iowa","kansas","kentucky",
            "louisi
