"""
Context-based Question Answering

Uses an LLM to answer questions based on given context text, with multi-provider support.

Main functions:
- ask(): Answer a question based on context

Dependencies:
- core.llm.model: Unified LLM dispatch
"""

from core.llm.model import llm_call


def ask(text: str, question: str, llm_provider: str = "azure") -> str:
    """
    Answer a question based on context text.

    Args:
        text: Context text to use for answering
        question: The question to answer
        llm_provider: LLM provider ("azure", "openai", or "openrouter")

    Returns:
        Answer string
    """
    # Prompt crafted to avoid triggering Azure content filters (removed instructions that could be flagged as jailbreak)
    prompt = (
        "Based on the context provided below, answer the question. "
        "If the answer is not in the context, respond with 'None'.\n\n"
        f"Context:\n{text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    return llm_call(prompt, llm_provider=llm_provider)


if __name__ == "__main__":
    context = "Microsoft Corporation is a technology company."
    question = "What is Microsoft?"
    answer = ask(context, question)
    print(f"Answer: {answer}")
