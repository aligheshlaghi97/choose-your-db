"""
FastAPI backend for database recommendation using vector search with Google Gemini embeddings.

This application helps users find the best database from "7 Databases in 7 Weeks"
by matching their requirements against a vector database of database descriptions.
"""

import os
from typing import Dict, List

import google.generativeai as genai
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from db_loader import configure_gemini, load_databases_to_qdrant

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Database Recommendation API",
    description="AI-powered database recommendation system using vector search with Google Gemini embeddings",
    version="0.0.1",
)
origins = [
    "http://localhost:3000",
    "https://aligheshlaghi97.github.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # which origins can call your API
    allow_credentials=True,
    allow_methods=["*"],  # allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # allow all headers
)

# Initialize Qdrant client (in-memory)
qdrant_client = QdrantClient(":memory:")

# Initialize Gemini (will be configured during startup)
gemini_configured: bool = False

# Configuration
USE_LLM_EXPLANATIONS = os.getenv("USE_LLM_EXPLANATIONS", "true").lower() == "true"

SAMPLE_QUESTIONS = {
    "q1": "What is the main type of data you want to store?",
    "q2": "How important are relationships (connections) between your data?",
    "q3": "Do you need your database to handle very large scale (terabytes/petabytes) of data?",
    "q4": "How critical is strong consistency (all users always see the latest data)?",
    "q5": "How important is high availability (system keeps running even if parts fail)?",
    "q6": "Do you need flexible or evolving data structures (not fixed schema)?",
    "q7": "Do you want extremely fast response times (sub-millisecond)?",
    "q8": "Do you need offline or unreliable-network support with automatic sync later?",
    "q9": "What are your main use cases?",
}

SAMPLE_ANSWERS = {
    "q1": [
        "Structured (tables, rows, columns)",  # PostgreSQL
        "Semi-structured (JSON, flexible fields)",  # PostgreSQL, MongoDB, CouchDB, DynamoDB
        "Graph-like (networks, relationships)",  # Neo4j
        "Key-value or cache style",  # Redis, DynamoDB
        "Column-family (huge sparse tables)",  # HBase
    ],
    "q2": [
        "Very important (e.g., social networks, fraud detection)",  # Neo4j
        "Somewhat important",  # PostgreSQL, MongoDB
        "Not important",  # Redis, HBase, DynamoDB
    ],
    "q3": [
        "Yes, I expect petabytes of data",  # HBase, DynamoDB
        "Yes, but more like terabytes",  # PostgreSQL, MongoDB
        "No, only gigabytes or less",  # Redis, CouchDB, Neo4j
    ],
    "q4": [
        "Must always be consistent (banking, financial apps)",  # PostgreSQL, Neo4j, HBase
        "Can tolerate some delays (eventual consistency is fine)",  # MongoDB, CouchDB, DynamoDB
        "Not important for my case",
    ],
    "q5": [
        "Always available is critical (uptime must not drop)",  # DynamoDB, CouchDB
        "Availability is important, but consistency is more important",  # PostgreSQL, Neo4j, HBase
        "I don’t really care much",
    ],
    "q6": [
        "Yes, data structures will change often",  # MongoDB, CouchDB, DynamoDB
        "Somewhat, but mostly structured",  # PostgreSQL
        "No, fixed schema is fine",  # HBase, Neo4j
    ],
    "q7": [
        "Yes, I need sub-millisecond performance",  # Redis
        "Fast but not ultra-critical",  # DynamoDB, PostgreSQL
        "Speed is not my top concern",
    ],
    "q8": [
        "Yes, my users/devices may be offline but should sync later",  # CouchDB
        "No, always online access is expected",
    ],
    "q9": [
        "Transactional systems (banking, payments)",  # PostgreSQL
        "Big data logs / time-series / sensors",  # HBase
        "Web/mobile apps with flexible data",  # MongoDB
        "Offline-first / sync across devices",  # CouchDB
        "Social networks / recommendation engines",  # Neo4j
        "Gaming leaderboards / IoT / high-scale apps",  # DynamoDB
        "Caching / real-time analytics / sessions",  # Redis
    ],
}


# Pydantic models for request/response
class RecommendationRequest(BaseModel):
    answers: Dict[str, List[str]] = Field(
        ...,
        description="User answers to the 9 questions",
        example={
            "q1": ["Structured (tables, rows, columns)"],
            "q2": ["Very important (e.g., social networks, fraud detection)"],
            "q3": ["Yes, but more like terabytes"],
            "q4": ["Must always be consistent (banking, financial apps)"],
            "q5": ["Availability is important, but consistency is more important"],
            "q6": ["Somewhat, but mostly structured"],
            "q7": ["Fast but not ultra-critical"],
            "q8": ["No, always online access is expected"],
            "q9": ["Transactional systems (banking, payments)"],
        },
    )


class DatabaseRecommendation(BaseModel):
    name: str
    score: float
    explanation: str


class RecommendationResponse(BaseModel):
    recommendations: List[DatabaseRecommendation]
    query_summary: str


@app.on_event("startup")
async def startup_event():
    """Initialize the application and load database descriptions into Qdrant."""
    global gemini_configured

    print("Starting up Database Recommendation API...")

    try:
        # Configure Gemini API
        configure_gemini()
        gemini_configured = True
        print("Google Gemini API configured successfully")

        # Load database descriptions into Qdrant
        count = load_databases_to_qdrant(qdrant_client)
        print(f"Successfully loaded {count} database descriptions")

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set GOOGLE_API_KEY environment variable")
        raise e
    except Exception as e:
        print(f"Error during startup: {e}")
        raise e


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Database Recommendation API",
        "version": "0.0.1",
        "features": {
            "vector_search": "Google Gemini embedding-001",
            "llm_explanations": USE_LLM_EXPLANATIONS,
            "llm_model": "gemini-1.5-flash" if USE_LLM_EXPLANATIONS else "disabled",
        },
        "endpoints": {
            "POST /recommend": "Get database recommendations based on your requirements",
            "GET /questions": "Get the list of questions and possible answers",
        },
    }


@app.get("/questions")
async def get_questions():
    """Get the list of questions and possible answers for the recommendation system."""
    return {
        "questions": SAMPLE_QUESTIONS,
        "answer_choices": SAMPLE_ANSWERS,
        "description": "Answer these questions to get personalized database recommendations",
    }


def adjust_scores(db_scores, user_answers):
    # strong signal from q1
    if "Graph-like (networks, relationships)" in user_answers.get("q1"):
        db_scores["Neo4j"] += 0.10

    # latency preference
    if "Yes, I need sub-millisecond performance" in user_answers.get("q7"):
        db_scores["Redis"] += 0.10

    # offline sync
    if "Yes, my users/devices may be offline but should sync later" in user_answers.get(
        "q8"
    ):
        db_scores["CouchDB"] += 0.10

    return db_scores


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_database(request: RecommendationRequest):
    """
    Get database recommendations based on user answers.

    This endpoint converts user answers into a text query, embeds it using Google Gemini,
    finds the most similar database descriptions using vector search, and optionally
    generates polished explanations using Gemini-1.5-flash.
    """
    if not gemini_configured:
        raise HTTPException(status_code=500, detail="Google Gemini API not configured")

    try:
        # Convert user answers into a descriptive text query
        query_text = _build_query_from_answers(request.answers)

        # Generate embedding for the query using Gemini
        query_embedding = _get_embedding_safe(query_text)

        # Search for similar databases in Qdrant
        search_results = qdrant_client.search(
            collection_name="databases",
            query_vector=query_embedding,
            limit=7,  # Get more results to allow for score adjustments
            with_payload=True,
        )

        # Apply custom score adjustments based on user answers
        db_scores = {}
        for result in search_results:
            db_scores[result.payload["name"]] = result.score

        # Adjust scores based on specific user answers
        adjusted_scores = adjust_scores(db_scores, request.answers)

        # Sort by adjusted scores and take top 3
        sorted_databases = sorted(
            adjusted_scores.items(), key=lambda x: x[1], reverse=True
        )[:3]

        # Format recommendations
        recommendations = []
        for db_name, adjusted_score in sorted_databases:
            # Find the original search result for this database
            original_result = next(
                r for r in search_results if r.payload["name"] == db_name
            )

            # Generate explanation (LLM if enabled, fallback to basic explanation)
            explanation = await _generate_explanation(
                db_name,
                adjusted_score,  # Use adjusted score for explanation
                query_text,
                original_result.payload["description"],
            )

            recommendation = DatabaseRecommendation(
                name=db_name,
                score=adjusted_score,  # Use adjusted score
                explanation=explanation,
            )
            recommendations.append(recommendation)

        return RecommendationResponse(
            recommendations=recommendations, query_summary=query_text
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating recommendations: {str(e)}"
        )


def _get_embedding_safe(text: str) -> List[float]:
    """
    Get embedding with error handling and fallback.

    Args:
        text: Text to embed

    Returns:
        List of floats representing the embedding vector

    Raises:
        HTTPException: If embedding generation fails
    """
    try:
        embedding = genai.embed_content(model="models/embedding-001", content=text)
        return embedding["embedding"]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate embedding: {str(e)}"
        )


async def _generate_explanation(
    db_name: str, score: float, query: str, db_description: str
) -> str:
    """
    Generate an explanation for why a database was recommended.

    If LLM explanations are enabled, uses Gemini-1.5-flash for polished output.
    Otherwise, falls back to basic template-based explanations.

    Args:
        db_name: Name of the recommended database
        score: Similarity score from vector search
        query: Original user query
        db_description: Database description from knowledge base

    Returns:
        Explanation string
    """
    if not USE_LLM_EXPLANATIONS or not gemini_configured:
        return _generate_basic_explanation(db_name, score, query)

    try:
        # Use Gemini-1.5-flash to generate polished explanation
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
        You are a database expert helping explain why a specific database was recommended.

        Database: {db_name}
        Database Description: {db_description}
        User Requirements: {query}
        Similarity Score: {score:.3f}

        Generate a concise, professional explanation (2-3 sentences) explaining why this database
        is a good match for the user's requirements. Focus on specific strengths and capabilities
        that align with their needs. Include the confidence level based on the similarity score.

        Format: Start with the database name, then explain the recommendation, end with confidence level.
        """

        response = model.generate_content(prompt)

        return response.text.strip()

    except Exception as e:
        print(f"LLM explanation generation failed: {e}")
        # Fallback to basic explanation
        return _generate_basic_explanation(db_name, score, query)


def _generate_basic_explanation(db_name: str, score: float, query: str) -> str:
    """
    Generate a basic explanation when LLM is not available or fails.

    Args:
        db_name: Name of the recommended database
        score: Similarity score from vector search
        query: Original user query

    Returns:
        Basic explanation string
    """
    confidence = "high" if score > 0.7 else "moderate" if score > 0.5 else "low"

    explanations = {
        "PostgreSQL": "PostgreSQL is recommended for its robust ACID compliance, strong SQL standards adherence, and hybrid capabilities handling both relational and JSON data with excellent extensibility.",
        "HBase": "HBase is recommended for its distributed architecture built on Hadoop, ability to handle billions of rows with real-time read/write access, and excellent scalability for big data workloads.",
        "MongoDB": "MongoDB is recommended for its flexible document-oriented design, horizontal scaling capabilities, and excellent performance with unstructured or semi-structured data requiring rapid development.",
        "CouchDB": "CouchDB is recommended for its unique multi-master replication, offline-first capabilities, RESTful HTTP API, and eventual consistency model perfect for distributed collaboration systems.",
        "Neo4j": "Neo4j is recommended for its native graph database design, efficient relationship traversal using Cypher query language, and excellent performance for applications with complex entity connections.",
        "DynamoDB": "DynamoDB is recommended for its fully managed AWS service, predictable performance at any scale, high availability across multiple zones, and perfect fit for internet-scale applications.",
        "Redis": "Redis is recommended for its exceptional in-memory performance, support for multiple data structures, versatility as cache/database/message broker, and sub-millisecond response times.",
    }

    base_explanation = explanations.get(
        db_name, f"{db_name} is recommended based on your requirements."
    )
    return f"{base_explanation} Confidence: {confidence} (score: {score:.3f})"


def _build_query_from_answers(answers: Dict[str, List[str]]) -> str:
    """
    Convert user answers into a descriptive text query for vector search.

    Args:
        answers: Dictionary of question IDs to answer lists

    Returns:
        Descriptive text query
    """
    query_parts = []

    # Map question IDs to meaningful descriptions
    question_mapping = {
        "q1": "data type",
        "q2": "relationship importance",
        "q3": "data scale",
        "q4": "consistency requirements",
        "q5": "availability vs consistency priority",
        "q6": "schema flexibility needs",
        "q7": "performance requirements",
        "q8": "offline support needs",
        "q9": "use case",
    }

    for q_id, answer_list in answers.items():
        if q_id in question_mapping:
            question_desc = question_mapping[q_id]
            answers_str = ", ".join(answer_list)
            query_parts.append(f"{question_desc}: {answers_str}")

    # Combine all parts into a coherent query
    query = f"I need a database for an application with {'; '.join(query_parts)}. "
    query += "The database should be well-suited for these requirements and provide good performance and reliability."

    return query


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
