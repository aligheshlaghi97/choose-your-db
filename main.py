"""
FastAPI backend for database recommendation using vector search with Google Gemini embeddings.

This application helps users find the best database from "7 Databases in 7 Weeks"
by matching their requirements against a vector database of database descriptions.
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import google.generativeai as genai
import uvicorn
from dotenv import load_dotenv

from db_loader import load_databases_to_qdrant, configure_gemini

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Database Recommendation API",
    description="AI-powered database recommendation system using vector search with Google Gemini embeddings",
    version="0.0.1",
)

# Initialize Qdrant client (in-memory)
qdrant_client = QdrantClient(":memory:")

# Initialize Gemini (will be configured during startup)
gemini_configured: bool = False

# Configuration
USE_LLM_EXPLANATIONS = os.getenv("USE_LLM_EXPLANATIONS", "true").lower() == "true"

# Sample questions that users would answer
SAMPLE_QUESTIONS = {
    "q1": "What type of data are you primarily working with?",
    "q2": "What are your performance requirements?",
    "q3": "What is your expected data volume?",
    "q4": "What consistency guarantees do you need?",
    "q5": "What is your deployment environment?",
    "q6": "What is your team's expertise level?",
    "q7": "What is your budget constraint?",
    "q8": "What is your time-to-market requirement?",
    "q9": "What integration requirements do you have?",
    "q10": "What is your scaling strategy?",
}

# Sample answer choices for each question
SAMPLE_ANSWERS = {
    "q1": ["structured", "unstructured", "semi-structured", "graph", "time-series"],
    "q2": ["high-speed", "moderate", "high-throughput", "real-time", "batch"],
    "q3": ["small", "medium", "large", "massive", "growing"],
    "q4": ["strong", "eventual", "weak", "custom", "none"],
    "q5": ["cloud", "on-premise", "hybrid", "edge", "distributed"],
    "q6": ["beginner", "intermediate", "expert", "mixed", "consulting"],
    "q7": ["low", "medium", "high", "enterprise", "open-source"],
    "q8": ["immediate", "quick", "moderate", "planned", "flexible"],
    "q9": ["simple", "moderate", "complex", "legacy", "modern"],
    "q10": ["vertical", "horizontal", "auto", "manual", "hybrid"],
}


# Pydantic models for request/response
class RecommendationRequest(BaseModel):
    answers: Dict[str, List[str]] = Field(
        ...,
        description="User answers to the 10 questions",
        example={
            "q1": ["structured", "semi-structured"],
            "q2": ["high-speed"],
            "q3": ["large"],
            "q4": ["strong"],
            "q5": ["cloud"],
            "q6": ["intermediate"],
            "q7": ["medium"],
            "q8": ["quick"],
            "q9": ["moderate"],
            "q10": ["horizontal"],
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
            limit=3,  # Return top 3 recommendations
            with_payload=True,
        )

        # Format recommendations
        recommendations = []
        for result in search_results:
            # Generate explanation (LLM if enabled, fallback to basic explanation)
            explanation = await _generate_explanation(
                result.payload["name"],
                result.score,
                query_text,
                result.payload["description"],
            )

            recommendation = DatabaseRecommendation(
                name=result.payload["name"],
                score=result.score,
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
        "q2": "performance requirements",
        "q3": "data volume",
        "q4": "consistency needs",
        "q5": "deployment environment",
        "q6": "expertise level",
        "q7": "budget constraints",
        "q8": "time requirements",
        "q9": "integration needs",
        "q10": "scaling approach",
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
