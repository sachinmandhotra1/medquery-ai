# MedQuery AI - Medical Context Query System

MedQuery AI is an intelligent medical research assistant powered by AI that provides access to clinical trials and medical publications. The system uses advanced natural language processing to provide context-aware responses backed by medical literature.

## Features

- **Clinical Trials Access**: Search and analyze data from thousands of clinical trials
- **Publication Search**: Access medical publications and research papers with intelligent summarization
- **Smart Context**: AI-powered responses backed by relevant medical context and citations
- **Interactive Chat Interface**: User-friendly chat interface for querying medical information
- **Conversation History**: Track and review past queries and responses

## Tech Stack

- **Backend**: FastAPI
- **Database**: PostgreSQL with Vector extension
- **AI/LLM Integration**: OpenAI API
- **Frontend**: HTML/Jinja2 Templates
- **Connection Pooling**: psycopg2
- **Environment Management**: python-dotenv

## Prerequisites

- Python 3.10+
- PostgreSQL 13+ with Vector extension
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd medquery-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
OPENAI_API_KEY=your_openai_api_key
MODEL=gpt-3.5-turbo  # or your preferred model
DB_SCHEMA=your_schema_name
```

## Database Setup

The system automatically initializes the required database schema and tables on startup. The schema includes:

- Clinical trials feed table
- Publications feed table
- Vector embeddings for semantic search
- Required PostgreSQL extensions (pg_trgm, vector)

### Initialize Embeddings

After setting up the database, you need to generate the initial embeddings:

```bash
python -m app.utils.embedding_updater
```

This will create vector embeddings for all entries in the clinical trials and publications tables, which are required for semantic search functionality.

## Project Structure
```
app/
├── init.py
├── main.py # FastAPI application entry point
├── config.py # Configuration settings
├── database.py # Database management
├── mcp_server.py # MCP (Model Context Protocol) server
├── utils/
│ ├── init.py
│ ├── postgres.py # PostgreSQL connection manager
│ └── schema_init.py # Database schema initialization
├── models/
│ └── init.py
├── static/ # Static files
└── templates/ # HTML templates
├── base.html
├── home.html
├── chat.html
└── history.html
```


## Running the Application

1. Start the server:
```bash
uvicorn app.main:app --reload
```

2. Access the application at `http://localhost:8000`

## API Endpoints

- `GET /`: Home page
- `GET /chat`: Chat interface
- `GET /history`: Conversation history
- `POST /query`: Process queries and return AI responses
- `GET /health`: Health check endpoint
- `GET /mcp/context/{query}`: Fetch relevant medical context

## Database Connection Management

The system uses a connection pooling mechanism to efficiently manage database connections with the following features:

- Minimum connections: 1
- Maximum connections: 10
- Retry attempts: 3
- Retry delay: 2 seconds

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
