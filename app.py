import streamlit as st
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -----------------------------
# Sidebar Controls
# -----------------------------

k = st.sidebar.slider("Top K Results", 1, 5, 2)

embedding_model_name = st.sidebar.selectbox(
    "Embedding Model",
    [
        "all-MiniLM-L6-v2",
        "multi-qa-MiniLM-L6-cos-v1"
    ]
)

max_len = st.sidebar.slider(
    "Max Response Length",
    10,
    200,
    60
)

temperature = st.sidebar.slider(
    "Temperature",
    0.1,
    1.5,
    0.7
)
top_p = st.sidebar.slider(
    "Top P",
    0.1,
    1.0,
    0.9
)

# -----------------------------
# Load Embedding Model
# -----------------------------

embed_model = SentenceTransformer(embedding_model_name)

# -----------------------------
# Load Dataset
# -----------------------------

with open("data/faq.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

chunks = [line.strip() for line in lines if line.strip()]

# -----------------------------
# Create Embeddings
# -----------------------------

embeddings = embed_model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")

# -----------------------------
# Build FAISS Index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# -----------------------------
# Load LLM
# -----------------------------

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

# -----------------------------
# UI
# -----------------------------

st.title("College FAQ RAG Chatbot")

query = st.text_input("Ask a question:")

if query:

    # Query embedding
    query_embedding = embed_model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    # Retrieve docs
    distances, indices = index.search(query_embedding, k)

    retrieved_docs = [chunks[i] for i in indices[0]]

    context = "\n".join(retrieved_docs)

    # Prompt
    prompt = f"""
    You are a helpful and conversational college assistant chatbot.

    Answer the student's question in a detailed and engaging way.

    Use complete sentences and explain the information naturally.

    If possible, elaborate using the provided context.

    Context:
    {context}

    Student Question:
    {query}

    Detailed Answer:
    """

    # Generate answer
    response = generator(
    prompt,
    max_length=max_len,
    temperature=temperature,
    top_p=top_p,
    do_sample=True,
    top_k=50,
    repetition_penalty=1.2,
    num_return_sequences=1
    )


    answer = response[0]['generated_text']

    # Show answer
    st.subheader("Generated Answer")

    st.write(answer)

    # Show retrieved docs
    st.subheader("Retrieved Context")

    for doc in retrieved_docs:
        st.write("-", doc)