# 4_rag_answer.py

from retrieval import retrieve


def answer_question(query):

    retrieved = retrieve(query, k=3)
    context = "\n\n".join(retrieved)

    prompt = f"""
    Answer the question using only the context below.

    Context:
    {context}

    Question:
    {query}
    """

    # Replace this with Gemini / OpenAI call
    print("=== PROMPT SENT TO LLM ===")
    print(prompt)

    # Placeholder response
    return "LLM response goes here."
