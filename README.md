# Database Recommendation API

An AI-powered FastAPI backend that recommends databases from "7 Databases in 7 Weeks" using vector search with Google Gemini embeddings and optional Gemini-1.5-flash explanations.

🌐 **Live Demo**: [Test the app here](https://aligheshlaghi97.github.io/choose-your-db-frontend/)

## 🎯 Goal

This application helps users find the best database for their needs by matching their requirements against a knowledge base of database descriptions using Google's Gemini embedding-001 model and vector search with Qdrant.

## 🏗️ Architecture

- **Backend Framework**: FastAPI
- **Vector Database**: Qdrant (in-memory setup)
- **Embeddings**: Google Gemini `models/embedding-001`
- **Optional LLM**: Gemini-1.5-flash for polished explanations
- **Lightweight**: Designed for 1 vCPU, 1 GB RAM VMs (so few concurrent users expected)

## 🚀 Features

- **Smart Recommendations**: Uses Google Gemini embeddings and vector search to find the most similar database descriptions
- **10 Question Assessment**: Comprehensive questionnaire covering data type, performance, volume, consistency, deployment, expertise, budget, timeline, integration, scaling, and additional requirements
- **Top 3 Recommendations**: Returns ranked database suggestions with explanations
- **AI-Powered Explanations**: Optional Gemini-1.5-flash integration for polished, human-readable explanations
- **Fallback Support**: Graceful degradation to template-based explanations if LLM fails
- **Lightweight**: Runs entirely in memory, perfect for development and testing


## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aligheshlaghi97/choose-your-db.git
   cd choose-your-db
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Google Gemini API**:
   ```bash
   # Copy the example config file
   cp config.env.example .env
   
   # Edit .env and add your Google Gemini API key
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

## 🚀 Usage

### Try the Live Demo

🌐 [**Test the app online**](https://aligheshlaghi97.github.io/choose-your-db-frontend/)

### Starting the Server Locally

```bash
python main.py
```

The server will start on `http://localhost:8000` and automatically:
- Initialize the Google Gemini client
- Load database descriptions into Qdrant with Gemini embeddings

### API Endpoints

#### 1. Root Endpoint
- **GET** `/`
- Returns API information, features, and available endpoints

#### 2. Questions Endpoint
- **GET** `/questions`
- Returns the list of 10 questions and possible answer choices

#### 3. Recommendation Endpoint
- **POST** `/recommend`
- Accepts user answers and returns database recommendations with AI-powered explanations

### Example Request

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
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
      "q10": ["No, the 9 questions above cover everything I need"]
    }
  }'
```

### Example Response

```json
{
    "recommendations": [
        {
            "name": "HBase",
            "score": 0.7443209368182635,
            "explanation": "HBase is recommended due to its ability to handle petabyte-scale, sparse datasets with a column-family structure, perfectly aligning with the user's big data logs and sensor data use case.  Its strong consistency guarantees (CP system) meet the critical requirement for financial applications, prioritizing data integrity over immediate availability.  Confidence level: High (0.744)."
        },
        {
            "name": "MongoDB",
            "score": 0.7261291375207646,
            "explanation": "MongoDB, while a strong choice for many applications, is not the ideal solution for this specific use case.  Its AP characteristics, prioritizing availability over strong consistency, directly conflict with the user's stringent consistency requirements for a banking/financial application.  Therefore, a database offering stronger consistency guarantees would be a more suitable choice.  Confidence level: Low (0.726 similarity score indicates a poor match)."
        },
        {
            "name": "Neo4j",
            "score": 0.7174694917360691,
            "explanation": "Neo4j, while a powerful graph database excelling in relationship-rich data, is not the optimal choice for this use case.  Its strength in managing interconnected data and high consistency are overshadowed by its limitations in handling petabyte-scale, sparse columnar data and its prioritization of consistency over availability, which conflicts with the user's needs for a highly available system for big data logs.  Confidence level: Low (0.717 similarity score indicates a poor match)."
        }
    ],
    "query_summary": "I need a database for an application with data type: Column-family (huge sparse tables); relationship importance: Not important; data scale: Yes, I expect petabytes of data; consistency requirements: Must always be consistent (banking, financial apps); availability vs consistency priority: Availability is important, but consistency is more important; schema flexibility needs: No, fixed schema is fine; performance requirements: Speed is not my top concern; offline support needs: No, always online access is expected; use case: Big data logs / time-series / sensors. The database should be well-suited for these requirements and provide good performance and reliability."
}
```

## 🧪 Testing

Run the test script to see the API in action:

```bash
python test_api.py
```

This will test all endpoints and show different recommendation scenarios.

## 🔧 Configuration

### Environment Variables

- **`GOOGLE_API_KEY`** (required): Your Google Gemini API key
- **`USE_LLM_EXPLANATIONS`** (optional): Enable/disable LLM explanations (default: true)

### Questions and Answers

The 10 questions cover:
1. **Data Type**: What is the main type of data you want to store?
2. **Relationships**: How important are relationships (connections) between your data?
3. **Scale**: Do you need your database to handle very large scale (terabytes/petabytes) of data?
4. **Consistency**: How critical is strong consistency (all users always see the latest data)?
5. **Availability**: How important is high availability (system keeps running even if parts fail)?
6. **Schema Flexibility**: Do you need flexible or evolving data structures (not fixed schema)?
7. **Performance**: Do you want extremely fast response times (sub-millisecond)?
8. **Offline Support**: Do you need offline or unreliable-network support with automatic sync later?
9. **Use Cases**: What are your main use cases?
10. **Additional Requirements**: Is there anything else you want to include that's missing in the 9 questions above?

### Database Descriptions

The system includes descriptions for 7 databases:
- PostgreSQL
- HBase
- MongoDB
- CouchDB
- Neo4j
- DynamoDB
- Redis

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
