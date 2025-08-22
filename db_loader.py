"""
Database loader utility for populating Qdrant vector database with database descriptions.

This module simulates loading database summaries from a PDF into a vector database.
In a real implementation, this would parse actual PDF content and chunk it appropriately.
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import google.generativeai as genai
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Mock database descriptions based on "7 Databases in 7 Weeks"
# In reality, these would come from parsing the actual PDF content
# Each description is 1-2 paragraphs summarizing key strengths
DATABASE_DESCRIPTIONS = [
    {
        "name": "PostgreSQL",
        "description": "PostgreSQL is a powerful, open-source object-relational database system with over 30 years of active development. It offers advanced features like complex queries, foreign keys, triggers, updatable views, and transactional integrity. PostgreSQL excels at handling complex queries and large datasets, making it ideal for applications requiring ACID compliance and sophisticated data relationships. It's particularly strong in analytical workloads and can handle both structured and semi-structured data through JSON support.",
    },
    {
        "name": "MongoDB",
        "description": "MongoDB is a document-oriented NoSQL database that stores data in flexible, JSON-like documents. It's designed for high availability and horizontal scaling, making it excellent for applications that need to handle large amounts of unstructured or semi-structured data. MongoDB shines in scenarios requiring rapid development, flexible schema evolution, and when you need to scale horizontally across multiple servers. It's particularly well-suited for content management systems, real-time analytics, and applications with rapidly changing data structures.",
    },
    {
        "name": "Redis",
        "description": "Redis is an in-memory data structure store that can be used as a database, cache, and message broker. It supports various data structures such as strings, hashes, lists, sets, and sorted sets. Redis excels at providing extremely fast read and write operations, making it perfect for caching, session storage, real-time analytics, and as a message queue. Its persistence options allow it to survive restarts while maintaining its speed advantages. Redis is ideal when you need sub-millisecond response times and can fit your working dataset in memory.",
    },
    {
        "name": "Neo4j",
        "description": "Neo4j is a native graph database that stores data in nodes and relationships, making it perfect for applications that need to traverse complex relationships between entities. It excels at social networks, recommendation engines, fraud detection, and any scenario where understanding connections between data points is crucial. Neo4j's Cypher query language is designed specifically for graph operations, making complex relationship queries intuitive and efficient. It's particularly strong when you need to answer questions about how different entities are connected and when you need to traverse multiple relationship hops efficiently.",
    },
    {
        "name": "Cassandra",
        "description": "Apache Cassandra is a distributed NoSQL database designed to handle large amounts of data across many commodity servers, providing high availability with no single point of failure. It's built for linear scalability and can handle massive amounts of data with high write throughput. Cassandra excels in write-heavy workloads, time-series data, and applications requiring global distribution. Its eventual consistency model makes it perfect for use cases where you can trade immediate consistency for availability and partition tolerance, such as logging systems, IoT data collection, and real-time analytics.",
    },
    {
        "name": "HBase",
        "description": "HBase is a distributed, scalable, big data store built on top of Apache Hadoop. It provides real-time read/write access to large datasets and is designed to handle billions of rows and millions of columns. HBase excels at random, real-time read/write access to big data and is particularly strong for applications requiring linear and modular scalability. It's built on the Hadoop Distributed File System (HDFS) and integrates well with other Hadoop ecosystem tools. HBase is ideal for sparse datasets, time-series data, and when you need to store and retrieve large amounts of data with predictable access patterns.",
    },
    {
        "name": "Riak",
        "description": "Riak is a distributed NoSQL database that emphasizes availability and fault tolerance. It's designed to provide high availability, fault tolerance, and operational simplicity. Riak excels in scenarios where you need guaranteed availability and can't afford downtime, making it perfect for critical applications like user session storage, user preferences, and real-time bidding systems. Its eventual consistency model and automatic data distribution make it highly resilient to network partitions and server failures. Riak is particularly well-suited for applications requiring high availability and when you need to add or remove nodes without downtime.",
    },
]


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
                size=768, distance=Distance.COSINE
            ),  # Gemini embedding-001 size
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
    Get embedding for text using Google Gemini's embedding-001 model.

    Args:
        text: Text to embed

    Returns:
        List of floats representing the embedding vector
    """
    try:
        embedding = genai.embed_content(model="models/embedding-001", content=text)
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


def get_database_descriptions():
    """
    Get the list of database descriptions for reference.

    Returns:
        List of database information dictionaries
    """
    return DATABASE_DESCRIPTIONS
