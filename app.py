from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import requests
from groq import Groq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

document_chunks = []
chunk_embeddings = []
embedding_model = None
conversation_history = {}
groq_client = None

def init_models():
    global embedding_model, groq_client
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    print("Models loaded!")

def extract_doc_id(url):
    patterns = [
        r'/document/d/([a-zA-Z0-9-_]+)',
        r'id=([a-zA-Z0-9-_]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def fetch_google_doc(doc_url):
    try:
        doc_id = extract_doc_id(doc_url)
        if not doc_id:
            return None, "Invalid Google Doc URL format"
        
        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
        response = requests.get(export_url, timeout=15)
        
        if response.status_code == 200:
            content = response.text.strip()
            if len(content) < 50:
                return None, "Document appears to be empty"
            return content, None
        elif response.status_code == 404:
            return None, "Document not found. Make sure it's publicly accessible."
        else:
            return None, f"Cannot access document. Ensure it's set to 'Anyone with link can view'"
    except Exception as e:
        return None, f"Error: {str(e)}"

def chunk_document(text, chunk_size=800, overlap=150):
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        if len(chunk_text.strip()) > 100:
            chunks.append({
                'text': chunk_text,
                'index': len(chunks),
                'section': f"Section {len(chunks) + 1}",
                'word_count': len(chunk_words)
            })
    
    return chunks

def create_embeddings(chunks):
    global embedding_model
    texts = [chunk['text'] for chunk in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    return embeddings

def retrieve_relevant_chunks(query, top_k=3):
    global document_chunks, chunk_embeddings, embedding_model
    
    if not document_chunks or chunk_embeddings is None:
        return []
    
    query_embedding = embedding_model.encode([query], show_progress_bar=False)[0]
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.25:
            results.append({
                'text': document_chunks[idx]['text'],
                'section': document_chunks[idx]['section'],
                'score': float(similarities[idx])
            })
    
    return results

def generate_answer(query, context_chunks, history):
    global groq_client
    
    try:
        if context_chunks:
            context = "\n\n".join([
                f"[{chunk['section']}] {chunk['text'][:600]}..."
                for chunk in context_chunks
            ])
        else:
            context = "No relevant information found."
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on document content. Rules: 1. ALWAYS cite sections (e.g., According to Section 2...). 2. If not in context, say This information isn't in the document. 3. Keep answers 2-4 sentences. 4. Be accurate - only use the context provided"
            }
        ]
        
        for msg in history[-6:]:
            messages.append(msg)
        
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        })
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=400,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/ingest', methods=['POST'])
def ingest_document():
    global document_chunks, chunk_embeddings
    
    data = request.json
    doc_url = data.get('url', '').strip()
    
    if not doc_url:
        return jsonify({'error': 'No URL provided'}), 400
    
    if 'docs.google.com' not in doc_url:
        return jsonify({'error': 'Please provide a valid Google Docs URL'}), 400
    
    print(f"Fetching: {doc_url}")
    content, error = fetch_google_doc(doc_url)
    
    if error:
        return jsonify({'error': error}), 400
    
    document_chunks = chunk_document(content)
    print(f"Created {len(document_chunks)} chunks")
    
    chunk_embeddings = create_embeddings(document_chunks)
    print("Embeddings created!")
    
    return jsonify({
        'status': 'success',
        'chunks': len(document_chunks),
        'message': f'Document processed into {len(document_chunks)} chunks'
    })

@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history
    
    data = request.json
    query = data.get('query', '').strip()
    session_id = data.get('session_id', 'default')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    if not document_chunks:
        return jsonify({'error': 'Please ingest a document first'}), 400
    
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    relevant_chunks = retrieve_relevant_chunks(query, top_k=3)
    answer = generate_answer(query, relevant_chunks, conversation_history[session_id])
    
    conversation_history[session_id].append({"role": "user", "content": query})
    conversation_history[session_id].append({"role": "assistant", "content": answer})
    
    if len(conversation_history[session_id]) > 10:
        conversation_history[session_id] = conversation_history[session_id][-10:]
    
    return jsonify({
        'answer': answer,
        'sources': [chunk['section'] for chunk in relevant_chunks]
    })

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    init_models()
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)