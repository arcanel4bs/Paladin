from flask import Flask, request, jsonify, render_template, send_from_directory
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import Dict, Any
import os
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from werkzeug.utils import secure_filename
import time
from uuid import uuid4
import json
from PyPDF2 import PdfReader
from docx import Document
from collections import deque
import threading
import logging
import tiktoken
import hashlib

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_remove_file(filepath):
    """Safely remove a file if it exists"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        logger.error(f"Error removing file {filepath}: {e}")

def cleanup_upload_folder():
    """Clean up any leftover files in the upload folder"""
    try:
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                safe_remove_file(filepath)
    except Exception as e:
        logger.error(f"Error cleaning upload folder: {e}")

app = Flask(__name__, static_folder='static')

# Configure app settings
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

# Ensure upload folder exists and is clean
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
cleanup_upload_folder()

DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'

# Initialize the Groq LLM
GROQ_LLM = ChatGroq(
    temperature=0.1,
    model="llama3-70b-8192",
    api_key=os.getenv('GROQ_API_KEY')
)

# Simple in-memory cache for file summaries
file_cache = {}

# Initialize conversation cache
conversation_cache = {}
MAX_CONVERSATIONS = 10
cache_lock = threading.Lock()

# Initialize tools
search_tool = DuckDuckGoSearchRun()

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
        
        # Always use Groq LLM for summarization
        summary = summarize_file_content(file_content)

        if summary and summary.strip():
            file_cache[file_hash] = summary
            return summary
        return "Could not generate a summary for the file."
        
    except Exception as e:
        logger.error(f"Error in cache_file_content: {e}")
        return "Error processing file content."

def summarize_file_content(file_content):
    """Summarize content using Groq LLM"""
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """Summarize the following content concisely and effectively.
        Focus on key points and maintain important details while reducing length.
        Ensure the summary is clear and well-structured."""),
        ("user", "{file_content}")
    ])
    chain = summary_prompt | GROQ_LLM | StrOutputParser()
    summary = chain.invoke({"file_content": file_content})
    return summary

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
    raw_search_results: str

# Add new function for search result summarization
def summarize_search_results(raw_results: str) -> str:
    """Summarizes search results to extract key details."""
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """As an AI assistant, extract and organize the most relevant information from these search results.
        Include:
        - Key facts, dates, and figures
        - Direct quotes from reliable sources
        - Historical context and background
        - Current developments and implications
        
        Format the information in a clear, structured manner."""),
        ("user", "{raw_results}")
    ])
    chain = summary_prompt | GROQ_LLM | StrOutputParser()
    return chain.invoke({"raw_results": raw_results})

# Add new formatting function after summarize_search_results
def format_response_markdown(response: str) -> str:
    """Formats the response in clean markdown."""
    format_prompt = ChatPromptTemplate.from_messages([
        ("system", """Format the given text into clean, readable markdown.
        
        Guidelines:
        - Use proper heading levels (# for main title, ## for sections, etc.)
        - Add line breaks between sections
        - Use bullet points or numbered lists where appropriate
        - Highlight key terms with bold or italics
        - Use blockquotes for important quotes
        - Add horizontal rules to separate major sections if needed
        - Preserve all factual content - only change formatting
        - Do not include any prefix or meta text about formatting
        - Start directly with the content
        
        Example Format:
        # Main Title
        
        ## Background
        
        Key information here...
        """),
        ("user", "{text}")
    ])
    
    chain = format_prompt | GROQ_LLM | StrOutputParser()
    formatted = chain.invoke({"text": response})
    
    # Remove any remaining formatting prefixes if they exist
    formatted = formatted.replace('Here is the formatted text in clean markdown:', '').strip()
    return formatted

# Create prompt templates
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to web search capabilities and file upload.
Analyze the user's input to determine if additional information from the web is needed.

Instructions:
1. Identify the core question and its components
2. Determine if answering requires:
   - Current events or news
   - Historical context
   - Statistical data
   - Expert opinions or quotes
3. If any of these are needed, conclude with 'Search Required: True'
4. If the question can be fully answered with existing context, conclude with 'Search Required: False'

Output Format:
Analysis: [Your detailed analysis]
Search Required: [True/False]"""),
    ("user", """Conversation History: {conversation_history}
Current User Input: {input_text}
File Summary (if provided): {file_summary}
Web Search Results (if available): {search_results}""")
])

response_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a knowledgeable AI assistant.
When generating your response, please:

- Provide a detailed and informative answer to the user's query
- Incorporate relevant data, figures, dates, and direct quotes from the search results
- Offer contextual background to enhance understanding
- Present the information in a structured and coherent manner
- Use markdown formatting for better readability

Avoid:
- Leaving out important details
- Repeating information unnecessarily
- Providing personal opinions or unsupported statements"""),
    ("user", """Conversation History:
{conversation_history}

Based on this analysis:
{intermediate_result}

Search Results:
{search_results}

Generate a comprehensive response that fully addresses the user's question.""")
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
    """Performs web search, summarizes results, and updates state."""
    try:
        logger.info(f"Performing search for: {state['input_text']}")
        raw_results = search_tool.run(state["input_text"])
        state['raw_search_results'] = raw_results
        
        # Summarize the raw results
        summarized_results = summarize_search_results(raw_results)
        state['search_results'] = summarized_results
        
        logger.info(f"Raw Search Results: {raw_results[:200]}...")
        logger.info(f"Summarized Results: {summarized_results[:200]}...")
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        state['raw_search_results'] = "Search unavailable"
        state['search_results'] = "Search unavailable"
    return state

def generate_output(state: WorkflowState) -> WorkflowState:
    """Generates the final, detailed response."""
    try:
        logger.info("Generating final detailed response")
        # Generate initial response
        chain = response_prompt | GROQ_LLM | StrOutputParser()
        initial_result = chain.invoke({
            "intermediate_result": state["intermediate_result"],
            "conversation_history": state["conversation_history"],
            "file_summary": state.get("file_summary", ""),
            "search_results": state.get("search_results", ""),
        })
        
        # Format the response in markdown
        formatted_result = format_response_markdown(initial_result)
        state['final_result'] = formatted_result
        logger.info(f"Generated formatted response: {formatted_result[:200]}...")
        
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

def cleanup_old_conversations():
    """Remove old conversations if cache exceeds maximum size"""
    with cache_lock:
        if len(conversation_cache) > MAX_CONVERSATIONS:
            sorted_convos = sorted(conversation_cache.items(), 
                                 key=lambda x: x[1]['timestamp'])
            to_remove = len(conversation_cache) - MAX_CONVERSATIONS
            for i in range(to_remove):
                del conversation_cache[sorted_convos[i][0]]

def cache_conversation(conversation_id: str, message_data: dict):
    """Cache conversation with timestamp"""
    with cache_lock:
        if conversation_id not in conversation_cache:
            conversation_cache[conversation_id] = {
                'messages': deque(maxlen=5),
                'timestamp': time.time()
            }
        conversation_cache[conversation_id]['messages'].append(message_data)
        cleanup_old_conversations()

def get_conversation_history(conversation_id):
    """Get conversation history from cache"""
    if conversation_id not in conversation_cache:
        return ""
    
    history = []
    for msg in conversation_cache[conversation_id]['messages']:
        history.extend([
            f"User: {msg['user_input']}",
            f"Assistant: {msg['ai_response']}"
        ])
    return "\n".join(history)

def safe_workflow_invoke(inputs: Dict) -> Dict:
    """Safely invoke the workflow with error handling"""
    try:
        return app_workflow.invoke(inputs)
    except Exception as e:
        logger.error(f"Workflow error: {e}")
        return {
            "final_result": "I apologize, but I encountered an error processing your request.",
            "intermediate_result": str(e),
            "decision": "error",
            "search_results": ""
        }

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering home page: {e}")
        return jsonify({
            "status": "error",
            "message": "Error loading application",
            "error": str(e)
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.form.get('message', '')
        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        conversation_id = request.form.get('conversation_id')
        if not conversation_id:
            conversation_id = str(uuid4())

        start_time = time.time()

        # Get conversation history
        conversation_history = get_conversation_history(conversation_id)

        # Run the workflow
        inputs = {
            "input_text": user_input,
            "file_content": "",
            "file_summary": "",
            "search_results": "",
            "intermediate_result": "",
            "conversation_history": conversation_history,
            "conversation_id": conversation_id,
            "decision": "",
            "raw_search_results": "",
            "final_result": ""
        }

        output = safe_workflow_invoke(inputs)
        
        if not output.get("final_result"):
            logger.error("No response generated")
            return jsonify({"error": "No response generated"}), 500

        # Calculate latency
        latency = (time.time() - start_time) * 1000

        # Return response with debug information
        response_data = {
            "status": "success",
            "response": output["final_result"],
            "conversation_id": conversation_id,
            "latency": latency
        }

        if DEBUG_MODE:
            response_data.update({
                "debug": {
                    "intermediate_result": output.get("intermediate_result", ""),
                    "search_decision": output.get("decision", ""),
                    "search_results": output.get("search_results", ""),
                }
            })

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "conversation_id": conversation_id if 'conversation_id' in locals() else None
        }), 500

@app.route('/history', methods=['GET'])
def get_history():
    with cache_lock:
        history = []
        for conv_id, conv_data in conversation_cache.items():
            for msg in conv_data['messages']:
                history.append({
                    'user_input': msg['user_input'],
                    'ai_response': msg['ai_response'],
                    'timestamp': msg['timestamp']
                })
        return jsonify(sorted(history, key=lambda x: x['timestamp'], reverse=True)[:50])

@app.route('/static/<path:path>')
def serve_static(path):
    try:
        return send_from_directory('static', path)
    except Exception as e:
        logger.error(f"Error serving static file {path}: {e}")
        return jsonify({"error": "File not found"}), 404

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({
        "status": "error",
        "message": "Internal server error",
        "error": str(e)
    }), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True)