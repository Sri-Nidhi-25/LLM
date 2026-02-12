import pickle
import ollama
import math

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

with open("vector_db.pkl", "rb") as f:
    VECTOR_DB = pickle.load(f)

print(f"Loaded {len(VECTOR_DB)} stored embeddings")

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

def retrieve(query, top_n=5, threshold=0.75):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0]
    sims = []
    for chunk, label, emb in VECTOR_DB:
        score = cosine_similarity(query_embedding, emb)
        if score >= threshold:
            sims.append((chunk, label, score))
    sims.sort(key=lambda x: x[2], reverse=True)
    return sims[:top_n]


while True:
    input_query = input("\nAsk a question (or type exit): ")
    if input_query.lower() in ("exit", "quit"):
        print("Goodbye 👋")
        break

    retrieved = retrieve(input_query)

    # Show which source each chunk came from
    print("\nRetrieved knowledge:")
    for text, label, score in retrieved:
        print(f" - ({label}, sim:{score:.2f}) {text}")

    context_text = "\n".join(
        f"[{label.upper()}] {text}" for text, label, _ in retrieved
    )

    instruction_prompt = f"""
You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't invent new information.
{context_text}
"""

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": input_query},
        ],
        stream=True,
    )

    print("\nChatbot response:")
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
