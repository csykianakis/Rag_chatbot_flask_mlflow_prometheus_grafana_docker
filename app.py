import os
import tempfile
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, render_template, session, jsonify
from huggingface_hub import login
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from prometheus_flask_exporter import PrometheusMetrics
import mlflow
from uuid import uuid4

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")
metrics = PrometheusMetrics(app, path="/metrics")
# metrics = PrometheusMetrics(app=None, path='/metrics')

mlflow.set_tracking_uri("http://mlflow:5000")
# mlflow.set_tracking_uri("file:/tmp/mlruns")
mlflow.set_experiment("PDF_QA_Chat_Experiment")

hf_token = os.getenv("HF_TOKEN")
login(hf_token)

# LlamaIndex settings
Settings.llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    tokenizer_name="meta-llama/Llama-3.2-1B-Instruct",
    device_map="auto",
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.2},
    system_prompt="You are a helpful and friendly assistant who gives short, direct answers based on the uploaded PDF."
)

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=50)

ALLOWED_EXTENSIONS = {'pdf'}
document_cache = {}
memory_cache = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def build_query_engine_with_memory(file_stream, filename):
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = os.path.join(tmpdirname, filename)
        file_stream.seek(0)
        with open(tmp_path, 'wb') as f:
            f.write(file_stream.read())
        docs = SimpleDirectoryReader(tmpdirname).load_data()
        index = VectorStoreIndex.from_documents(docs)
        retriever = index.as_retriever()
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

        # Add response synthesizer for compact, concise answers
        response_synthesizer = get_response_synthesizer(response_mode="compact")

        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            memory=memory,
            response_synthesizer=response_synthesizer
        )
        return query_engine, memory

def refine_prompt(user_message):
    return f"Please provide a medium size, direct answer to the following question: {user_message}"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = Path(file.filename).name
    session['session_id'] = str(uuid4())

    query_engine, memory = build_query_engine_with_memory(file.stream, filename)
    document_cache[session['session_id']] = query_engine
    memory_cache[session['session_id']] = memory

    return jsonify({'message': f'File {filename} uploaded successfully.'})

@app.route('/chat', methods=['POST'])
def chat():
    if 'session_id' not in session:
        return jsonify({'error': 'No document uploaded.'}), 400

    data = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'error': 'Message is empty.'}), 400

    query_engine = document_cache.get(session['session_id'])
    if not query_engine:
        return jsonify({'error': 'Session expired or no document found.'}), 400

    refined_message = refine_prompt(user_message)

    start_time = datetime.now()
    response = query_engine.query(refined_message)
    end_time = datetime.now()

    with mlflow.start_run():
        mlflow.log_param("session_id", session['session_id'])
        mlflow.log_param("user_message", user_message)
        mlflow.log_metric("response_time", (end_time - start_time).total_seconds())
        mlflow.log_text(str(response), "chat_response.txt")

    return jsonify({'response': str(response)})

if __name__ == '__main__':
    print("\nRegistered Routes:")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint:25s} -> {rule.rule}")
    app.run(host='0.0.0.0', port=8000, debug=True)


