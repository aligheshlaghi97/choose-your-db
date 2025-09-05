"""
Test script for the Database Recommendation API.

This script demonstrates how to use the API endpoints and shows example requests.
Run this after starting the FastAPI server.

Note: You need to set GOOGLE_API_KEY environment variable for the API to work.
"""

import os

import requests
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
        if data["features"]["llm_explanations"]:
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
            "q1": [
                "Structured (tables, rows, columns)",
                "Semi-structured (JSON, flexible fields)",
            ],
            "q2": ["Very important (e.g., social networks, fraud detection)"],
            "q3": ["Yes, but more like terabytes"],
            "q4": ["Must always be consistent (banking, financial apps)"],
            "q5": ["Always available is critical (uptime must not drop)"],
            "q6": ["Yes, data structures will change often"],
            "q7": ["Fast but not ultra-critical"],
            "q8": ["No, always online access is expected"],
            "q9": ["Transactional systems (banking, payments)"],
            "q10": ["I want to handle very huge data but not sure how much is that"],
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

            for i, rec in enumerate(data["recommendations"], 1):
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
                "q1": ["Key-value or cache style"],
                "q2": ["Not important"],
                "q3": ["No, only gigabytes or less"],
                "q4": ["Not important for my case"],
                "q5": ["I don't really care much"],
                "q6": ["No, fixed schema is fine"],
                "q7": ["Yes, I need sub-millisecond performance"],
                "q8": ["No, always online access is expected"],
                "q9": ["Caching / real-time analytics / sessions"],
                "q10": ["Need real-time data processing capabilities"],
            },
        },
        {
            "name": "Graph Database Scenario",
            "answers": {
                "q1": ["Graph-like (networks, relationships)"],
                "q2": ["Very important (e.g., social networks, fraud detection)"],
                "q3": ["No, only gigabytes or less"],
                "q4": ["Must always be consistent (banking, financial apps)"],
                "q5": ["Availability is important, but consistency is more important"],
                "q6": ["No, fixed schema is fine"],
                "q7": ["Fast but not ultra-critical"],
                "q8": ["No, always online access is expected"],
                "q9": ["Social networks / recommendation engines"],
                "q10": ["Must be open source and have a strong community"],
            },
        },
        {
            "name": "High-Scale Gaming/IoT Scenario (DynamoDB)",
            "answers": {
                "q1": ["Key-value or cache style"],
                "q2": ["Not important"],
                "q3": ["Yes, I expect petabytes of data"],
                "q4": ["Can tolerate some delays (eventual consistency is fine)"],
                "q5": ["Always available is critical (uptime must not drop)"],
                "q6": ["Yes, data structures will change often"],
                "q7": ["Fast but not ultra-critical"],
                "q8": ["No, always online access is expected"],
                "q9": ["Gaming leaderboards / IoT / high-scale apps"],
                "q10": ["Need geographic distribution across regions"],
            },
        },
        {
            "name": "Big Data Analytics Scenario (HBase)",
            "answers": {
                "q1": ["Column-family (huge sparse tables)"],
                "q2": ["Not important"],
                "q3": ["Yes, I expect petabytes of data"],
                "q4": ["Must always be consistent (banking, financial apps)"],
                "q5": ["Availability is important, but consistency is more important"],
                "q6": ["No, fixed schema is fine"],
                "q7": ["Speed is not my top concern"],
                "q8": ["No, always online access is expected"],
                "q9": ["Big data logs / time-series / sensors"],
                "q10": ["Want to minimize operational overhead"],
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
                for rec in data["recommendations"]:
                    print(
                        f"Recommendation: {rec['name']} (score: {rec['score']:.3f})"
                    )
                # top_rec = data["recommendations"][0]
                # print(
                #     f"Top recommendation: {top_rec['name']} (score: {top_rec['score']:.3f})"
                #     f"Explanation: {top_rec['explanation']}"
                # )
            else:
                print(f"Error: {response.status_code}")

        except Exception as e:
            print(f"Error: {e}")


def test_q10_free_text():
    """Test q10 free-text functionality with various inputs."""
    print("=== Testing Q10 Free-Text Functionality ===")
    
    test_cases = [
        {
            "name": "Huge Data Scale Question",
            "q10_text": "I want to handle very huge data but not sure how much is that"
        },
        {
            "name": "Custom Business Requirements",
            "q10_text": "Need to integrate with our existing microservices architecture and support real-time data streaming"
        },
        {
            "name": "Cost and Performance Balance",
            "q10_text": "Looking for a cost-effective solution that can scale from startup to enterprise level"
        },
        {
            "name": "Security and Compliance",
            "q10_text": "Must comply with GDPR and SOC2 requirements with built-in encryption at rest"
        }
    ]
    
    # Base answers for all test cases
    base_answers = {
        "q1": ["Semi-structured (JSON, flexible fields)"],
        "q2": ["Somewhat important"],
        "q3": ["Yes, but more like terabytes"],
        "q4": ["Can tolerate some delays (eventual consistency is fine)"],
        "q5": ["Availability is important, but consistency is more important"],
        "q6": ["Yes, data structures will change often"],
        "q7": ["Fast but not ultra-critical"],
        "q8": ["No, always online access is expected"],
        "q9": ["Web/mobile apps with flexible data"],
    }
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print(f"Q10 Input: {test_case['q10_text']}")
        
        # Add q10 to base answers
        test_answers = base_answers.copy()
        test_answers["q10"] = [test_case["q10_text"]]
        
        try:
            response = requests.post(
                f"{BASE_URL}/recommend",
                json={"answers": test_answers},
                headers={"Content-Type": "application/json"},
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"Query includes Q10: {'additional requirements' in data['query_summary']}")
                print(f"Top recommendation: {data['recommendations'][0]['name']} (score: {data['recommendations'][0]['score']:.3f})")
            else:
                print(f"Error: {response.status_code}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print()


def test_error_handling():
    """Test error handling with invalid requests."""
    print("=== Testing Error Handling ===")

    # Test with missing answers
    invalid_request = {
        "answers": {
            "q1": ["Structured (tables, rows, columns)"],
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
        print(
            "The API may not work properly without proper Google Gemini configuration."
        )
        print()

    # Test all endpoints
    test_root_endpoint()
    test_questions_endpoint()
    test_recommendation_endpoint()
    test_different_scenarios()
    test_q10_free_text()
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
    print('        -d \'{"answers": {"q1": ["structured"], "q2": ["high-speed"], "q10": ["custom requirements"]}}\'')
    print()
    print("Note: The API requires a valid Google Gemini API key to function.")
    print("Get your API key from: https://makersuite.google.com/app/apikey")
