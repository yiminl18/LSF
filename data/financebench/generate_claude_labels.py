import json
import time
import anthropic

BASE = "/Users/yiminglin/Documents/Codebase/LSF/data/financebench"
MODEL = "claude-sonnet-4-6"


def main():
    client = anthropic.Anthropic()

    with open(f"{BASE}/sample.txt") as f:
        first_file = f.readline().strip()

    with open(f"{BASE}/queries.txt") as f:
        queries = [line.strip() for line in f if line.strip()]

    txt_name = first_file.replace(".pdf", ".txt")
    with open(f"{BASE}/text/{txt_name}") as f:
        doc_text = f.read()

    print(f"File: {first_file}")
    print(f"Doc length: {len(doc_text)} chars, {len(queries)} questions\n")

    result = {first_file: {}}

    for i, question in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {question[:70]}...")
        start = time.time()

        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            output_config={"effort": "high"},
            system=[{
                "type": "text",
                "text": (
                    "You are a financial document analyst. "
                    "Answer each question about the document below concisely and accurately. "
                    "If the answer is a list, provide all items. "
                    "If information is not present in the document, respond with exactly the word: None\n\n"
                    f"DOCUMENT:\n{doc_text}"
                ),
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": question}],
        )

        latency = round(time.time() - start, 3)
        answer_text = next(b.text for b in response.content if b.type == "text")
        answer_value = None if answer_text.strip().lower() == "none" else answer_text

        result[first_file][question] = {
            "answer": answer_value,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency": latency,
        }
        print(f"  {answer_text[:80]}  ({latency}s, in={response.usage.input_tokens}, out={response.usage.output_tokens})")

    out_path = f"{BASE}/claude_label.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
