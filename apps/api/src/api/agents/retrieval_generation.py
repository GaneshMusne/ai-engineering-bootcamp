import os
from functools import lru_cache

from google import genai

COLLECTION_NAME = "Amazon-items-minilm-l6-v2"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def get_embedding(text):
    return get_embedding_model().encode(text, normalize_embeddings=True).tolist()

def retrieve_data(query, qdrant_client, k=5):

    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["description"])
        similarity_scores.append(result.score)
        retrieved_context_ratings.append(result.payload["average_rating"])

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings
    }

def process_context(context):

    formatted_context = ""

    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context

def build_prompt(preprocessed_context, question):

    prompt = f"""
You are a shopping assistant that can answer questions about the products in stock.

You will be given a question and a list of context.

Instructions:
- Answer the question based on the provided context only.
- Never use word context and refer to it as the available products.
- Do not use markdown formatting.

Context:
{preprocessed_context}

Question:
{question}
"""

    return prompt

def generate_answer(prompt):
    with genai.Client(api_key=os.getenv("GOOGLE_API_KEY")) as client:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
    if not response.text:
        raise RuntimeError("Gemini returned no text. Check the response for blocked content.")
    return response.text

def rag_pipeline(question, qdrant_client, top_k=5):
    retrieved_context = retrieve_data(question, qdrant_client, top_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    return generate_answer(prompt)
