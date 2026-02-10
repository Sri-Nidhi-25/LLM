import pickle
import ollama

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"

dataset = []
with open("cat-facts.txt", "r", encoding="utf-8") as file:
    dataset = file.readlines()

print(f"Loaded {len(dataset)} entries")

VECTOR_DB = []

def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)["embeddings"][0]
    VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    print(f"Added chunk {i+1}/{len(dataset)}")

# Save the built vector DB
with open("vector_db.pkl", "wb") as f:
    pickle.dump(VECTOR_DB, f)

print("Vector DB built and saved 🎉")
