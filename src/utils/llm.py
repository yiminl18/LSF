from gpt_4o_azure import gpt_4o_azure


def ask(text: str, question: str, key_path: str) -> str:
    """
    Ask a question based on the given context text using GPT-5.1.
    
    Args:
        text: Context text to use for answering
        question: The question to ask
        key_path: Path to the API key file (default: '/Users/evier/Documents/gpt-4o.txt')
    
    Returns:
        The answer string
    """
    prompt = (
        "You may think briefly before answering, but only output the final answer. "
        "If the answer cannot be inferred from the context, return None.\n\n"
        f"Context:\n{text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    
    answer = gpt_4o_azure(prompt, key_path=key_path)
    return answer


if __name__ == "__main__":
    # Example usage
    context = "Microsoft Corporation is a technology company."
    question = "What is Microsoft?"
    answer = ask(context, question)
    print(f"Answer: {answer}")

