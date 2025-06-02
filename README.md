# MedTechONE AI Agent

An intelligent documentation assistant built to help users navigate and understand MedTechONE's documentation. The agent uses advanced AI techniques to provide accurate, context-aware answers to questions about MedTechONE's systems and processes.

## Features

- Intelligent documentation processing and understanding
- Vector database storage with Supabase for efficient retrieval
- Semantic search using OpenAI embeddings
- RAG-based question answering with context awareness
- Support for code block and technical content preservation
- Mobile-friendly Streamlit UI for easy access
- Assessment mode for evaluating answer quality
- Support for both general queries and specific documentation sections

## Prerequisites

- Python 3.9+
- Supabase account and database
- OpenAI API key
- Streamlit (for web interface)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/brandondavies/MedTechONE-AI-agent.git
cd MedTechONE-AI-agent
```

2. Install dependencies (recommended to use a Python virtual environment):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your API keys and preferences:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   LLM_MODEL=gpt-4  # or your preferred OpenAI model
   ```

## Usage

### Database Setup

Execute the SQL commands in `site_pages.sql` to:
1. Create the necessary tables
2. Enable vector similarity search
3. Set up Row Level Security policies

In Supabase, go to the "SQL Editor" tab and paste in the SQL from `site_pages.sql`. Then click "Run".

### Process Documentation

To process and store documentation in the vector database:

```bash
python process_pdf_docs.py
```

This will:
1. Process PDF documentation
2. Split content into meaningful chunks
3. Generate embeddings and store in Supabase

### Web Interface

To start the interactive web interface:

```bash
streamlit run streamlit_ui.py
```

The interface will be available at `http://localhost:8501`

## Features

### Assessment Mode
The interface includes an assessment mode toggle that helps evaluate the quality and accuracy of the AI's responses. This is particularly useful for:
- Verifying technical accuracy
- Checking response completeness
- Ensuring proper context usage

### Mobile Optimization
The interface is optimized for mobile devices with:
- Responsive design
- Touch-friendly controls
- Readable text sizing
- Efficient space usage

## Project Structure

- `streamlit_ui.py`: Main web interface
- `MedTechONE_AI_Expert.py`: Core AI agent implementation
- `process_pdf_docs.py`: Documentation processor
- `site_pages.sql`: Database schema and setup
- `requirements.txt`: Project dependencies
- `assets/`: Static assets and resources

## Error Handling

The system includes robust error handling for:
- API rate limits
- Database connection issues
- Embedding generation errors
- Invalid content processing
- Network failures

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is proprietary and confidential. All rights reserved.
