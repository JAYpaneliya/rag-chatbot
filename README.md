cat > README.md << EOF
# RAG Document Chatbot

AI-powered chatbot that reads Google Docs and answers questions using Retrieval-Augmented Generation (RAG).

## Features
- 📄 Ingests content from Google Docs
- 🔍 Semantic search with vector embeddings
- 🤖 AI-powered answers with citations
- 💬 Multi-turn conversation support
- 🚀 Free deployment using Groq API

## Local Setup

1. Clone the repository
2. Create virtual environment: \`python -m venv venv\`
3. Activate: \`source venv/bin/activate\` (Mac/Linux) or \`venv\\Scripts\\activate\` (Windows)
4. Install dependencies: \`pip install -r requirements.txt\`
5. Create \`.env\` file with: \`GROQ_API_KEY=your_key_here\`
6. Run: \`python app.py\`
7. Open: http://localhost:8000

## Usage

1. Paste a public Google Doc URL
2. Click "Load Document"
3. Ask questions about the content

## Tech Stack
- Backend: Flask, Groq API (Llama 3.1)
- Embeddings: Sentence Transformers
- Vector Search: Scikit-learn
- Frontend: Vanilla JavaScript
EOF