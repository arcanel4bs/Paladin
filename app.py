from flask import Flask, request, jsonify, render_template
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import Dict, Any
import os
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from db.models import db, ChatMessage
import time
from uuid import uuid4
import json
from werkzeug.utils import secure_filename
import hashlib
from flask_migrate import Migrate
import google.generativeai as genai
import tiktoken
import logging
from PyPDF2 import PdfReader
from docx import Document

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db.init_app(app)
migrate = Migrate(app, db)

# Initialize tools
search_tool = DuckDuckGoSearchRun()

# Initialize the Groq LLM
GROQ_LLM = ChatGroq(
    temperature=0.1,
    model="llama3-70b-8192",
    api_key=os.getenv('GROQ_API_KEY')
)

# Initialize Gemini API client
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Simple in-memory cache for file summaries
file_cache = {}

DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_token_count(text):
    """Calculate number of tokens in text using tiktoken"""
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    return len(tokenizer.encode(text))

def cache_file_content(file_content):
    """Cache and summarize file content"""
    try:
        file_hash = hashlib.md5(file_content.encode()).hexdigest()
        if file_hash in file_cache:
            return file_cache[file_hash]
        
        # Use token count instead of word count for model selection
        if calculate_token_count(file_content) > 1000:
            summary = summarize_file_with_gemini(file_content)
        else:
            summary = summarize_file_content(file_content)

        if summary and summary.strip():
            file_cache[file_hash] = summary
            return summary
        return "Could not generate a summary for the file."
        
    except Exception as e:
        logger.error(f"Error in cache_file_content: {e}")
        return "Error processing file content."

def summarize_file_content(file_content):
    # Existing summarization using Groq LLM
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following content concisely:"),
        ("user", "{file_content}")
    ])
    chain = summary_prompt | GROQ_LLM | StrOutputParser()
    summary = chain.invoke({"file_content": file_content})
    return summary

def summarize_file_with_gemini(file_content):
    try:
        prompt = f"Summarize the following content concisely:\n\n{file_content}"
        response = genai.generate_text(
            model="models/chat-bison-001",
            prompt=prompt,
            temperature=0.5,
            max_output_tokens=512,
        )
        summary = response.result
        return summary
    except Exception as e:
        print(f"Error summarizing with Gemini: {e}")
        return "Could not summarize the file content."

# Define our workflow state
class WorkflowState(TypedDict):
    input_text: str
    file_content: str
    file_summary: str
    search_results: str
    intermediate_result: str
    final_result: str
    conversation_history: str
    conversation_id: str
    decision: str

# Create prompt templates
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to web search capabilities.
Analyze the user's input and determine if additional information from the web is needed.

Instructions:
1. Identify the core question in the user's input.
2. Determine if answering the question requires current or external information not provided.
3. If so, conclude with 'Search Required: True'.
4. If not, conclude with 'Search Required: False'.

Output Format:
Reasoning: [Your reasoning here]
Search Required: [True/False]"""),
    ("user", """Conversation History: {conversation_history}
Current User Input: {input_text}
File Summary (if provided): {file_summary}
Web Search Results (if available): {search_results}""")
])

response_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
When generating your response, consider the following:
- Conversation history
- File summary (if provided)
- Search results (if available)

Provide clear, concise answers that directly address the user's question.
Avoid repeating information already present in the conversation."""),
    ("user", """Conversation History: {conversation_history}
Based on this analysis: {intermediate_result}

Search Results:
{search_results}

Generate a helpful response that incorporates any relevant information from the search results.""")
])

# Define workflow nodes
def process_input(state: WorkflowState) -> Dict[str, Any]:
    """Analyzes the input text and file summary"""
    try:
        combined_input = state["input_text"]
        if state.get("file_summary"):
            combined_input += f"\nFile Summary: {state['file_summary']}"

        chain = analysis_prompt | GROQ_LLM | StrOutputParser()
        result = chain.invoke({
            "input_text": combined_input,
            "file_summary": state.get("file_summary", ""),
            "search_results": "",
            "conversation_history": state["conversation_history"]
        })
        
        logger.info(f"Intermediate Result: {result}")
        
    except Exception as e:
        logger.error(f"Error in process_input: {e}")
        result = "Error analyzing input. Search Required: False"

    return {
        "intermediate_result": result,
        "conversation_history": state["conversation_history"],
        "conversation_id": state["conversation_id"],
        "file_summary": state.get("file_summary", "")
    }

def should_search(state: WorkflowState) -> WorkflowState:
    """Determines if web search is needed based on analysis"""
    try:
        if "Search Required: True" in state["intermediate_result"]:
            state['decision'] = 'search'
        else:
            state['decision'] = 'generate'
        logger.info(f"Search decision: {state['decision']}")
        
    except Exception as e:
        logger.error(f"Error in should_search: {e}")
        state['decision'] = 'generate'
    return state

def perform_search(state: WorkflowState) -> WorkflowState:
    """Performs web search if needed"""
    try:
        logger.info(f"Performing search for: {state['input_text']}")
        search_results = search_tool.run(state["input_text"])
        state['search_results'] = search_results
        logger.info(f"Search Results: {search_results[:200]}...")  # Log first 200 chars
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        state['search_results'] = "Search unavailable"
    return state

def generate_output(state: WorkflowState) -> WorkflowState:
    """Generates the final response."""
    try:
        logger.info("Generating final response")
        chain = response_prompt | GROQ_LLM | StrOutputParser()
        result = chain.invoke({
            "intermediate_result": state["intermediate_result"],
            "conversation_history": state["conversation_history"],
            "file_summary": state.get("file_summary", ""),
            "search_results": state.get("search_results", "")
        })
        state['final_result'] = result
        logger.info(f"Generated response: {result[:200]}...")  # Log first 200 chars
        
    except Exception as e:
        logger.error(f"Error in generate_output: {e}")
        state['final_result'] = "I apologize, but I encountered an error generating the response."
    return state

# Create and compile workflow
workflow = StateGraph(WorkflowState)

# Add all nodes to the workflow
workflow.add_node("process_input", process_input)
workflow.add_node("should_search", should_search)
workflow.add_node("perform_search", perform_search)
workflow.add_node("generate_output", generate_output)

# Define edges between nodes
workflow.add_edge("process_input", "should_search")

# Define the routing function
def route_from_should_search(state: WorkflowState) -> str:
    if state['decision'] == 'search':
        return 'perform_search'
    else:
        return 'generate_output'

# Add conditional edges using the routing function
workflow.add_conditional_edges(
    "should_search",
    route_from_should_search,
    ["perform_search", "generate_output"]
)

# Add edges for the rest of the workflow
workflow.add_edge("perform_search", "generate_output")
workflow.add_edge("generate_output", END)

# Set the entry point of the workflow
workflow.set_entry_point("process_input")

# Compile the workflow
app_workflow = workflow.compile()

def get_conversation_history(conversation_id):
    messages = ChatMessage.query.filter_by(conversation_id=conversation_id)\
        .order_by(ChatMessage.timestamp.desc())\
        .limit(5)\
        .all()
    
    history = []
    for msg in reversed(messages):
        history.extend([
            f"User: {msg.user_input}",
            f"Assistant: {msg.ai_response}"
        ])
    return "\n".join(history)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get('message', '')
    conversation_id = request.form.get('conversation_id')
    if not conversation_id:
        conversation_id = str(uuid4())
    
    file_content = ""
    file_name = ""
    file_summary = ""
    
    if 'file' in request.files:
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                file_content = ''
                file_name = filename
                extension = os.path.splitext(filename)[1].lower()

                if extension == '.txt':
                    with open(filepath, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                elif extension == '.pdf':
                    reader = PdfReader(filepath)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            file_content += text
                elif extension in ['.doc', '.docx']:
                    doc = Document(filepath)
                    for para in doc.paragraphs:
                        file_content += para.text + '\n'
                else:
                    logger.warning(f"Unsupported file type: {extension}")
                    file_content = None

                if file_content and file_content.strip():
                    file_summary = cache_file_content(file_content)
                else:
                    file_summary = "Could not read the file content."

            except Exception as e:
                logger.error(f"Error processing file: {e}")
                file_summary = "Could not process the file."
            finally:
                # Clean up the uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
    
    if not user_input and not file_content:
        return jsonify({"error": "No message or file provided"}), 400

    start_time = time.time()
    
    # Get conversation history
    conversation_history = get_conversation_history(conversation_id)
    
    # Run the workflow
    inputs = {
        "input_text": user_input,
        "file_content": "",
        "file_summary": file_summary,
        "search_results": "",
        "intermediate_result": "",
        "conversation_history": conversation_history,
        "conversation_id": conversation_id
    }
    
    output = app_workflow.invoke(inputs)
    
    # Calculate latency
    latency = (time.time() - start_time) * 1000
    
    # Store in database
    chat_message = ChatMessage(
        user_input=user_input,
        ai_response=output["final_result"],
        search_results=output.get("search_results", ""),
        file_content=file_content,
        file_name=file_name,
        file_summary=file_summary,
        latency=latency,
        conversation_id=conversation_id
    )
    db.session.add(chat_message)
    db.session.commit()
    
    response_data = {
        "response": output["final_result"],
        "file_summary": file_summary if file_summary else None,
        "latency": latency,
        "conversation_id": conversation_id
    }
    
    # Add debug information if DEBUG_MODE is enabled
    if DEBUG_MODE:
        response_data.update({
            "debug": {
                "intermediate_result": output.get("intermediate_result", ""),
                "search_decision": output.get("decision", ""),
                "search_results": output.get("search_results", ""),
            }
        })
    
    return jsonify(response_data)

@app.route('/history', methods=['GET'])
def get_history():
    messages = ChatMessage.query.order_by(ChatMessage.timestamp.desc()).limit(50).all()
    return jsonify([message.to_dict() for message in messages])

if __name__ == '__main__':
    app.run(debug=True)