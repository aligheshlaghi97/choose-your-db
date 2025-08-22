"""
Test script for the Database Recommendation API.

This script demonstrates how to use the API endpoints and shows example requests.
Run this after starting the FastAPI server.

Note: You need to set GOOGLE_API_KEY environment variable for the API to work.
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API base URL
BASE_URL = "http://localhost:8000"

def check_gemini_config():
    """Check if Google Gemini API key is configured."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        print("⚠️  WARNING: GOOGLE_API_KEY not configured!")
        print("Please set your Google Gemini API key in a .env file:")
        print("GOOGLE_API_KEY=your_actual_api_key_here")
        print()
        return False
    return True

def test_root_endpoint():
    """Test the root endpoint."""
    print("=== Testing Root Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"API Version: {data['version']}")
        print(f"Vector Search: {data['features']['vector_search']}")
        print(f"LLM Explanations: {data['features']['llm_explanations']}")
        if data['features']['llm_explanations']:
            print(f"LLM Model: {data['features']['llm_model']}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def test_questions_endpoint():
    """Test the questions endpoint."""
    print("=== Testing Questions Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/questions")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Number of questions: {len(data['questions'])}")
        print(f"Sample question: {data['questions']['q1']}")
        print(f"Sample answers: {data['answer_choices']['q1']}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def test_recommendation_endpoint():
    """Test the recommendation endpoint with sample data."""
    print("=== Testing Recommendation Endpoint ===")
    
    # Sample request data
    sample_request = {
        "answers": {
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
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/recommend",
            json=sample_request,
            headers={"Content-Type": "application/json"},
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Query Summary: {data['query_summary']}")
            print(f"Number of recommendations: {len(data['recommendations'])}")
            
            for i, rec in enumerate(data['recommendations'], 1):
                print(f"\nRecommendation {i}:")
                print(f"  Database: {rec['name']}")
                print(f"  Score: {rec['score']:.3f}")
                print(f"  Explanation: {rec['explanation']}")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")
    print()

def test_different_scenarios():
    """Test different scenarios to show variety in recommendations."""
    print("=== Testing Different Scenarios ===")
    
    scenarios = [
        {
            "name": "High-Performance Cache Scenario",
            "answers": {
                "q1": ["unstructured"],
                "q2": ["high-speed"],
                "q3": ["medium"],
                "q4": ["weak"],
                "q5": ["cloud"],
                "q6": ["intermediate"],
                "q7": ["low"],
                "q8": ["immediate"],
                "q9": ["simple"],
                "q10": ["vertical"],
            },
        },
        {
            "name": "Graph Database Scenario",
            "answers": {
                "q1": ["graph"],
                "q2": ["moderate"],
                "q3": ["medium"],
                "q4": ["strong"],
                "q5": ["on-premise"],
                "q6": ["expert"],
                "q7": ["medium"],
                "q8": ["planned"],
                "q9": ["complex"],
                "q10": ["horizontal"],
            },
        },
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        try:
            response = requests.post(
                f"{BASE_URL}/recommend",
                json={"answers": scenario["answers"]},
                headers={"Content-Type": "application/json"},
            )
            
            if response.status_code == 200:
                data = response.json()
                top_rec = data['recommendations'][0]
                print(f"Top recommendation: {top_rec['name']} (score: {top_rec['score']:.3f})")
            else:
                print(f"Error: {response.status_code}")
                
        except Exception as e:
            print(f"Error: {e}")

def test_error_handling():
    """Test error handling with invalid requests."""
    print("=== Testing Error Handling ===")
    
    # Test with missing answers
    invalid_request = {
        "answers": {
            "q1": ["structured"],
            # Missing other questions
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/recommend",
            json=invalid_request,
            headers={"Content-Type": "application/json"},
        )
        print(f"Invalid request status: {response.status_code}")
        if response.status_code != 200:
            print(f"Error response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    print()

if __name__ == "__main__":
    print("Database Recommendation API Test Script")
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print("=" * 50)
    
    # Check configuration
    if not check_gemini_config():
        print("The API may not work properly without proper Google Gemini configuration.")
        print()
    
    # Test all endpoints
    test_root_endpoint()
    test_questions_endpoint()
    test_recommendation_endpoint()
    test_different_scenarios()
    test_error_handling()
    
    print("\nTest completed!")
    print("\nTo run the tests manually:")
    print("1. Set your Google Gemini API key in a .env file:")
    print("   GOOGLE_API_KEY=your_actual_api_key_here")
    print("2. Start the server: python main.py")
    print("3. Run this test script: python test_api.py")
    print("4. Or test manually with curl:")
    print("   curl -X POST http://localhost:8000/recommend \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"answers\": {\"q1\": [\"structured\"], \"q2\": [\"high-speed\"]}}'")
    print()
    print("Note: The API requires a valid Google Gemini API key to function.")
    print("Get your API key from: https://makersuite.google.com/app/apikey")

