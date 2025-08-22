# Database Recommendation API

An AI-powered FastAPI backend that recommends databases from "7 Databases in 7 Weeks" using vector search with Google Gemini embeddings and optional Gemini-1.5-flash explanations.

## 🎯 Goal

This application helps users find the best database for their needs by matching their requirements against a knowledge base of database descriptions using Google's Gemini embedding-001 model and vector search with Qdrant.

## 🏗️ Architecture

- **Backend Framework**: FastAPI
- **Vector Database**: Qdrant (in-memory setup)
- **Embeddings**: Google Gemini `models/embedding-001`
- **Optional LLM**: Gemini-1.5-flash for polished explanations
- **Lightweight**: Designed for 1 OCPU, 1 GB RAM VMs

## 🚀 Features

- **Smart Recommendations**: Uses Google Gemini embeddings and vector search to find the most similar database descriptions
- **10 Question Assessment**: Comprehensive questionnaire covering data type, performance, volume, consistency, deployment, expertise, budget, timeline, integration, and scaling
- **Top 3 Recommendations**: Returns ranked database suggestions with explanations
- **AI-Powered Explanations**: Optional Gemini-1.5-flash integration for polished, human-readable explanations
- **Fallback Support**: Graceful degradation to template-based explanations if LLM fails
- **Lightweight**: Runs entirely in memory, perfect for development and testing

## 📁 Project Structure

```
choose-your-db/
├── main.py              # FastAPI application with endpoints
├── db_loader.py         # Database loading utility for Qdrant with Google Gemini
├── test_api.py          # Test script to demonstrate API usage
├── requirements.txt     # Python dependencies
├── config.env.example   # Environment variables template
└── README.md           # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
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

## 🔑 Google Gemini API Setup

1. **Get an API key** from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Create a `.env` file** in the project root:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   USE_LLM_EXPLANATIONS=true
   ```
3. **Optional**: Set `USE_LLM_EXPLANATIONS=false` to disable LLM explanations and use template-based ones

## 🚀 Usage

### Starting the Server

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
      "q1": ["structured", "semi-structured"],
      "q2": ["high-speed"],
      "q3": ["large"],
      "q4": ["strong"],
      "q5": ["cloud"],
      "q6": ["intermediate"],
      "q7": ["medium"],
      "q8": ["quick"],
      "q9": ["moderate"],
      "q10": ["horizontal"]
    }
  }'
```

### Example Response

```json
{
  "recommendations": [
    {
      "name": "PostgreSQL",
      "score": 0.823,
      "explanation": "PostgreSQL is recommended for its strong ACID compliance, complex query support, and excellent handling of structured data with JSON capabilities. It's particularly well-suited for applications requiring data integrity and sophisticated relationships, making it an ideal choice for your cloud-based, horizontally scalable system. Confidence: high (score: 0.823)"
    },
    {
      "name": "MongoDB",
      "score": 0.756,
      "explanation": "MongoDB is recommended for its flexible schema design and horizontal scaling capabilities, which align perfectly with your need for rapid development and cloud deployment. Its excellent performance with semi-structured data and ability to handle large datasets make it a strong contender for your requirements. Confidence: high (score: 0.756)"
    }
  ],
  "query_summary": "I need a database for an application with data type: structured, semi-structured; performance requirements: high-speed; data volume: large; consistency needs: strong; deployment environment: cloud; expertise level: intermediate; budget constraints: medium; time requirements: quick; integration needs: moderate; scaling approach: horizontal. The database should be well-suited for these requirements and provide good performance and reliability."
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
1. **Data Type**: structured, unstructured, semi-structured, graph, time-series
2. **Performance**: high-speed, moderate, high-throughput, real-time, batch
3. **Data Volume**: small, medium, large, massive, growing
4. **Consistency**: strong, eventual, weak, custom, none
5. **Deployment**: cloud, on-premise, hybrid, edge, distributed
6. **Expertise**: beginner, intermediate, expert, mixed, consulting
7. **Budget**: low, medium, high, enterprise, open-source
8. **Timeline**: immediate, quick, moderate, planned, flexible
9. **Integration**: simple, moderate, complex, legacy, modern
10. **Scaling**: vertical, horizontal, auto, manual, hybrid

### Database Descriptions

The system includes descriptions for 7 databases:
- PostgreSQL
- MongoDB
- Redis
- Neo4j
- Cassandra
- HBase
- Riak

## 🔮 Extending the System

### Adding Real PDF Data

1. **Replace mock data** in `db_loader.py` with real PDF parsing
2. **Implement text chunking** for longer documents
3. **Add metadata** like page numbers, sections, etc.

### Adding More Databases

1. **Extend** `DATABASE_DESCRIPTIONS` in `db_loader.py`
2. **Add explanations** in the `_generate_basic_explanation` function
3. **Retrain embeddings** if needed

### Customizing Questions

1. **Modify** `SAMPLE_QUESTIONS` and `SAMPLE_ANSWERS` in `main.py`
2. **Update** the question mapping in `_build_query_from_answers`
3. **Adjust** the query generation logic

## 🚨 Limitations

- **In-memory storage**: Data is lost on server restart
- **Mock data**: Uses simulated database descriptions
- **API dependency**: Requires Google Gemini API key and internet connection
- **Cost**: Google Gemini API calls incur costs (embeddings and optional LLM)

## 🛡️ Production Considerations

- **Persistent storage**: Use a real Qdrant instance
- **Real PDF parsing**: Implement proper document processing
- **API key management**: Use secure environment variable management
- **Rate limiting**: Implement Google Gemini API rate limiting
- **Caching**: Add Redis for response and embedding caching
- **Authentication**: Add API key or user authentication
- **Monitoring**: Add logging and metrics
- **Fallback strategies**: Implement multiple embedding providers

## 📚 Dependencies

- **FastAPI**: Modern web framework for building APIs
- **Qdrant**: Vector similarity search engine
- **Google Generative AI**: Official Google Gemini Python SDK for embeddings and chat completions
- **Uvicorn**: ASGI server for FastAPI
- **Pydantic**: Data validation using Python type annotations
- **python-dotenv**: Environment variable management

## 💰 Cost Considerations

- **Embeddings**: ~$0.0001 per 1K characters (Gemini embedding-001)
- **LLM Explanations**: ~$0.00015 per 1K characters (Gemini-1.5-flash)
- **Typical cost per recommendation**: $0.001-0.005
- **Monthly cost for 1000 users**: $1-5

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on concepts from "7 Databases in 7 Weeks"
- Uses the excellent Qdrant vector database
- Powered by Google's state-of-the-art Gemini embeddings and language models