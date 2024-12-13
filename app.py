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

load_dotenv()

app = Flask(__name__)

# Initialize tools
search_tool = DuckDuckGoSearchRun()

# Initialize the Groq LLM
GROQ_LLM = ChatGroq(
    temperature=0.1,
    model="llama3-70b-8192",
    api_key=os.getenv('GROQ_API_KEY')
)

# Define our workflow state
class WorkflowState(TypedDict):
    input_text: str
    search_results: str
    intermediate_result: str
    final_result: str

# Create prompt templates
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to web search capabilities.
    Analyze the user's input and determine if additional information from the web might be helpful."""),
    ("user", """User Input: {input_text}
    Web Search Results (if available): {search_results}
    
    Please analyze this information.""")
])

response_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that generates friendly responses."),
    ("user", "Based on this analysis: {intermediate_result}, generate a helpful response.")
])

# Define workflow nodes
def process_input(state: WorkflowState) -> Dict[str, Any]:
    """Analyzes the input text and performs web search if needed."""
    try:
        search_results = search_tool.run(state["input_text"])
    except Exception as e:
        search_results = "Search unavailable"
    
    chain = analysis_prompt | GROQ_LLM | StrOutputParser()
    result = chain.invoke({
        "input_text": state["input_text"],
        "search_results": search_results
    })
    
    return {
        "intermediate_result": result,
        "search_results": search_results
    }

def generate_output(state: WorkflowState) -> Dict[str, Any]:
    """Generates the final response."""
    chain = response_prompt | GROQ_LLM | StrOutputParser()
    result = chain.invoke({"intermediate_result": state["intermediate_result"]})
    return {"final_result": result}

# Create and compile workflow
workflow = StateGraph(WorkflowState)
workflow.add_node("process_input", process_input)
workflow.add_node("generate_output", generate_output)
workflow.add_edge("process_input", "generate_output")
workflow.add_edge("generate_output", END)
workflow.set_entry_point("process_input")

app_workflow = workflow.compile()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    # Run the workflow
    inputs = {"input_text": user_input, "search_results": "", "intermediate_result": ""}
    output = app_workflow.invoke(inputs)
    
    return jsonify({
        "response": output["final_result"]
    })

if __name__ == '__main__':
    app.run(debug=True)