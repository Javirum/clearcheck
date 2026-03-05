"""Seed the Pinecone knowledge base with curated misinformation patterns."""

import json
import time
from pathlib import Path

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MISINFO_PATH = DATA_DIR / "misinformation_patterns.json"
SCAM_PATH = DATA_DIR / "scam_patterns.json"


def load_patterns() -> list[dict]:
    patterns = []
    with open(MISINFO_PATH) as f:
        patterns.extend(json.load(f))
    if SCAM_PATH.exists():
        with open(SCAM_PATH) as f:
            patterns.extend(json.load(f))
    return patterns


def create_embedding(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def seed():
    patterns = load_patterns()
    print(f"Loaded {len(patterns)} patterns (misinformation + scam)")

    # Initialize clients
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status.get("ready"):
            time.sleep(1)
        print("Index created and ready")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists")

    index = pc.Index(PINECONE_INDEX_NAME)

    # Create embeddings and upsert
    vectors = []
    for pattern in patterns:
        # Combine claim and explanation for richer embedding
        embed_text = f"{pattern['claim']} {pattern['explanation']}"
        embedding = create_embedding(openai_client, embed_text)
        metadata = {
            "claim": pattern["claim"],
            "category": pattern["category"],
            "verdict": pattern["verdict"],
            "explanation": pattern["explanation"],
            "sources": ", ".join(pattern["sources"]),
            "date_added": pattern["date_added"],
        }
        if "red_flags" in pattern:
            metadata["red_flags"] = ", ".join(pattern["red_flags"])
        vectors.append({
            "id": pattern["id"],
            "values": embedding,
            "metadata": metadata,
        })
        print(f"  Embedded: {pattern['id']} - {pattern['claim'][:60]}...")

    # Upsert in batches
    batch_size = 10
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"  Upserted batch {i // batch_size + 1}")

    print(f"\nDone! {len(vectors)} vectors upserted to '{PINECONE_INDEX_NAME}'")
    stats = index.describe_index_stats()
    print(f"Index stats: {stats.total_vector_count} total vectors")


if __name__ == "__main__":
    seed()
