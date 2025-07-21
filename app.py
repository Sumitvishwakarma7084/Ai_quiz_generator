from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv
from flask import render_template
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from werkzeug.utils import secure_filename
import random

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL =  "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_and_embed(file_path):
    # 1. Load based on file type
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format")

    # 2. Load and split document
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    faiss_path = "faiss_index"

    # 3. Load existing index if available, else create new
    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(
            faiss_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(docs)
    else:
        vectorstore = FAISS.from_documents(docs, embedding_model)

    # 4. Save updated index
    vectorstore.save_local(faiss_path)

    return vectorstore

# ==========================
# Routes
# ==========================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/quiz')
def quiz_page():
    return render_template('quiz.html')

@app.route('/prompt')
def settings_page():
    return render_template('prompt.html')

@app.route('/quizzes')
def quizzes_page():
    return render_template('history.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        # Clear existing FAISS index if it exists
        faiss_path = "faiss_index"
        if os.path.exists(faiss_path):
            import shutil
            shutil.rmtree(faiss_path)
            
        file.save(filepath)
        load_and_embed(filepath)
        return jsonify({'message': 'File uploaded and processed successfully'}), 200
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

def parse_gemini_response(response):
    """Helper function to parse Gemini API response"""
    if response.status_code != 200:
        raise ValueError(f"Gemini API error: {response.text}")
    
    result = response.json()
    try:
        if not result.get("candidates"):
            raise ValueError("No candidates in response")
        
        content = result["candidates"][0].get("content", {})
        if not content:
            raise ValueError("No content in candidate")
            
        parts = content.get("parts", [])
        if not parts or not isinstance(parts, list):
            raise ValueError("Invalid parts format")
            
        first_part = parts[0]
        if "text" not in first_part:
            raise ValueError("No text in parts")
            
        return first_part["text"]
    except Exception as e:
        raise ValueError(f"Error parsing response: {str(e)}. Full response: {result}")

@app.route('/generate-quiz', methods=['POST'])
def generate_quiz():
    try:
        data = request.get_json()
        difficulty = data.get("difficulty", "Medium")
        question_type = data.get("questionType", "Multiple Choice")
        question_count = int(data.get("questionCount", 5))
        topic = data.get("topic", "")

        # Load FAISS vectorstore
        vectorstore = FAISS.load_local(
            "faiss_index",
            embedding_model,
            allow_dangerous_deserialization=True
        )
        
        # Use topic as query if provided, otherwise general query
        query = topic if topic else "Generate a relevant quiz from this document"
        docs = vectorstore.similarity_search(query, k=5)

        if not docs:
            return jsonify({"error": "No relevant content found in document."}), 400

        content = "\n".join([doc.page_content for doc in docs])
        if not content.strip():
            return jsonify({"error": "Empty content fetched from document."}), 400

        # Enhanced prompt with randomization
        prompt = f"""
        Based on the following content, generate a {difficulty.lower()} level quiz with exactly {question_count} {question_type.lower()} questions.
        The questions should be varied and cover different aspects of the content.
        
        IMPORTANT:
        - Generate EXACTLY {question_count} questions
        - Each question should be distinct and test different knowledge points
        - Include explanations for answers where possible
        - Format should be consistent and machine-readable
        
        For multiple choice questions use this format:
        Q1: [question text]?
        a) [option 1]
        b) [option 2]
        c) [option 3]
        d) [option 4]
        Answer: [correct letter]
        Explanation: [brief explanation]
        
        For true/false questions:
        Q1: [statement]?
        a) True
        b) False
        Answer: [True/False]
        Explanation: [brief explanation]
        
        For fill in the blank questions:
        Q1: [statement with _____ blank]?
        Answer: [correct answer]
        Explanation: [brief explanation]
        
        Content to base quiz on:
        {content}
        
        Now generate exactly {question_count} {question_type.lower()} questions:
        """

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.9,  # Higher temperature for more variety
                "maxOutputTokens": 2000
            }
        }

        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        output_text = parse_gemini_response(response)

        return jsonify({
            "quiz": output_text,
            "meta": {
                "difficulty": difficulty,
                "questionType": question_type,
                "questionCount": question_count,
                "source": "document"
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/generate-custom-quiz', methods=['POST'])
def generate_topic_quiz():
    try:
        data = request.get_json()
        topic = data.get("topic")
        difficulty = data.get("difficulty", "Medium")
        question_type = data.get("questionType", "Multiple Choice")  
        question_count = int(data.get("questionCount", 5))

        if not topic:
            return jsonify({"error": "Topic is required"}), 400

        # Define format instructions based on question type
        format_examples = {
            "multiple choice": """
Q1: What is the capital of France?
a) London
b) Paris
c) Berlin
d) Madrid
Answer: b
Explanation: Paris has been the capital of France since 508 AD.
""",
            "true/false": """
Q1: The Earth is flat.
a) True
b) False
Answer: b
Explanation: Scientific evidence shows the Earth is an oblate spheroid.
""",
            "fill in the blank": """
Q1: The process by which plants make their own food is called _____.
Answer: photosynthesis
Explanation: Photosynthesis converts light energy into chemical energy.
"""
        }

        qtype_lower = question_type.lower()
        format_example = format_examples.get(qtype_lower, "")

        # Add some randomness to the prompt
        random_adjectives = ["comprehensive", "engaging", "thought-provoking", "challenging", "insightful"]
        random_adj = random.choice(random_adjectives)
        
        prompt = f"""
Generate a {random_adj} quiz about: {topic}

Requirements:
- Difficulty: {difficulty}
- Question type: {question_type}
- Number of questions: Exactly {question_count}
- Format must be consistent and machine-readable
- Questions should cover different aspects of the topic
- Include explanations for answers

Example format:
{format_example}

Now generate exactly {question_count} {question_type.lower()} questions about {topic} at {difficulty} difficulty level:

1. First question:
"""

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.9,  # Higher temperature for more variety
                "maxOutputTokens": 2000,
                "topP": 0.9
            }
        }

        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        output_text = parse_gemini_response(response)

        return jsonify({
            "quiz": output_text,
            "meta": {
                "difficulty": difficulty,
                "questionType": question_type,
                "questionCount": question_count,
                "topic": topic
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
