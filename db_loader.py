"""
Database loader utility for populating Qdrant vector database with database descriptions.

This module loads database summaries from markdown files into a vector database.
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import google.generativeai as genai
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database names to load from markdown files
DATABASE_NAMES = [
    "PostgreSQL",
    "HBase",
    "MongoDB",
    "CouchDB",
    "Neo4j",
    "DynamoDB",
    "Redis",
]


def load_database_descriptions():
    """
    Load database descriptions from markdown files in the descriptions directory.

    Returns:
        List of database information dictionaries with name and description
    """
    descriptions = []
    descriptions_dir = "descriptions"

    # Map database names to their file numbers
    db_file_mapping = {
        "PostgreSQL": "1-PostgreSQL.md",
        "HBase": "2-HBase.md",
        "MongoDB": "3-MongoDB.md",
        "CouchDB": "4-CouchDB.md",
        "Neo4j": "5-Neo4j.md",
        "DynamoDB": "6-DynamoDB.md",
        "Redis": "7-Redis.md",
    }

    for db_name in DATABASE_NAMES:
        filename = db_file_mapping.get(db_name)
        if filename:
            filepath = os.path.join(descriptions_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    description = file.read().strip()
                    descriptions.append({"name": db_name, "description": description})
                    print(f"Loaded description for {db_name}")
            except FileNotFoundError:
                print(f"Warning: Description file for {db_name} not found: {filepath}")
            except Exception as e:
                print(f"Error reading description for {db_name}: {e}")

    return descriptions


# Load database descriptions from files
DATABASE_DESCRIPTIONS = load_database_descriptions()


def create_qdrant_collection(client: QdrantClient, collection_name: str = "databases"):
    """
    Create a Qdrant collection for storing database descriptions.

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to create
    """
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=3072, distance=Distance.COSINE
            ),  # gemini-embedding-001 default dimension is 3072
        )
        print(f"Created collection: {collection_name}")
    except Exception as e:
        print(f"Collection {collection_name} already exists or error occurred: {e}")


def configure_gemini():
    """
    Configure Google Gemini API with the API key.

    Returns:
        None

    Raises:
        ValueError: If GOOGLE_API_KEY is not set
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")

    genai.configure(api_key=api_key)
    print("Google Gemini API configured successfully")


def get_embedding(text: str) -> list:
    """
    Get embedding for text using Google Gemini's gemini-embedding-001 model.

    Args:
        text: Text to embed

    Returns:
        List of floats representing the embedding vector
    """
    try:
        embedding = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document",
        )
        return embedding["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise e


def load_databases_to_qdrant(client: QdrantClient, collection_name: str = "databases"):
    """
    Load database descriptions into Qdrant with Google Gemini embeddings.

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to populate

    Returns:
        Number of databases loaded

    Raises:
        ValueError: If Google API key is not configured
    """
    # Configure Gemini API
    configure_gemini()

    # Create collection if it doesn't exist
    create_qdrant_collection(client, collection_name)

    # Prepare points for insertion
    points = []
    for db_info in DATABASE_DESCRIPTIONS:
        # Generate embedding for the database description using Gemini
        embedding = get_embedding(db_info["description"])

        # Create point structure
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"name": db_info["name"], "description": db_info["description"]},
        )
        points.append(point)

    # Insert all points into the collection
    client.upsert(collection_name=collection_name, points=points)

    print(f"Loaded {len(points)} database descriptions into Qdrant")
    return len(points)
