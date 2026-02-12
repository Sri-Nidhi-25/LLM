import os
import pickle
import ollama

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"

# Files paired with labels
source_files = [
    ("dog-facts.txt", "dogs"),
    ("cat-facts.txt", "cats"),
]

VECTOR_DB = []

print("Loading text files...")

# Build dataset by capturing both text AND its label
dataset = []
for filepath, label in source_files:
    if not os.path.exists(filepath):
        print(f"⚠️ File not found: {filepath} — skipping")
        continue

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        text = line.strip()
        if not text:
            continue
        # Bind the correct label here
        dataset.append((text, label))

print(f"Chunks to embed: {len(dataset)}")

def normalize(text):
    return text.lower().strip()

# Embed and add to vector DB
def add_to_db(text, label):
    out = ollama.embed(model=EMBEDDING_MODEL, input=normalize(text))
    embs = out.get("embeddings", [])
    if not embs:
        print(f"⚠️ Skipped (no embeddings): {text[:40]}…")
        return False
    VECTOR_DB.append((text, label, embs[0]))
    return True

for i, (text, label) in enumerate(dataset, start=1):
    success = add_to_db(text, label)
    tag = label if success else f"skipped {label}"
    print(f"{tag} — {i}/{len(dataset)}")

# Save the DB
with open("vector_db.pkl", "wb") as f:
    pickle.dump(VECTOR_DB, f)

print("Saved vector_db.pkl 🎉")

