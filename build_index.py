from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

print("Loading model...")

model = SentenceTransformer('all-MiniLM-L6-v2')

print("Reading dataset...")

with open("data/faq.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

chunks = [line.strip() for line in lines if line.strip()]

print("Chunks loaded:", chunks)

print("Generating embeddings...")

embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")

print("Creating FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Saving FAISS index...")

faiss.write_index(index, "faiss_index.bin")

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("SUCCESS!")