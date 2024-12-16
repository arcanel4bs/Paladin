# LangSearch - AI-Powered Search and Document Interface

![LangSearch Interface](/public/image1.png)

## Overview
LangSearch is a sophisticated AI chat interface that combines the power of Llama 3 (70B) with web search capabilities through LangGraph workflows. It features a cyberpunk-inspired laboratory interface designed for seamless human-AI interaction.

## Features
- 🤖 **Advanced LLM Integration**: Powered by Llama 3 70B through Groq's API
- 🔍 **Web Search Integration**: Real-time DuckDuckGo search capabilities
- 🔄 **Document Upload and Summarization**: Upload `.txt`, `.pdf`, `.doc`, and `.docx` files for summarization and analysis
- 🔄 **LangGraph Workflow**: Multi-step processing for enhanced response quality
- 🎨 **Sci-fi Laboratory UI**: Cyberpunk-inspired interface with real-time status indicators
- ⚡ **Performance Metrics**: Built-in latency tracking and system status monitoring

## Technical Stack
- **Frontend**: TailwindCSS, Custom CSS Animations
- **Backend**: Flask, LangChain, LangGraph
- **LLM**: Groq (Llama 3 70B)
- **Search**: DuckDuckGo Search API
- **File Processing**: Supports text, PDF, and Word documents

## Getting Started

### Prerequisites
- Python 3.8+
- Groq API Key
- Google Gemini API Key (for summarizing large files)
- DuckDuckGo Search capabilities
- Required Python packages: `PyPDF2`, `python-docx`, etc.

### Installation
1. Clone the repository:
```bash
git clone https://github.com/arcanel4bs/langsearch.git
cd langsearch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
GROQ_API_KEY=your_groq_api_key
```
```bash
GEMINI_API_KEY=your_gemini_api_key
```

4. Run database migrations:
```bash
flask db upgrade
```

5. Run the application:
```bash
python app.py
```

6. Access the application:

    Open your browser and navigate to ` http://127.0.0.1:5000`

## How It Works
The application uses a multi-stage LangGraph workflow:

1. **Input Processing**:
   - Analyzes user input and any uploaded file summaries.
   - Determines if a web search is needed for the response.

2. **Conditional Branching**:
   - If additional information is required, performs a web search using DuckDuckGo API.
   - Summarizes search results for inclusion in the response.

3. **Response Generation**:
   - Generates contextual responses using all available information:
     - Conversation history
     - File summaries
     - Web search results

## Document Upload and Summarization

- **Supported File Types**: `.txt`, `.pdf`, `.doc`, `.docx`
- **File Processing**:
  - **Text Files**: Read and summarized directly.
  - **PDF Files**: Extract text from pages using `PyPDF2`.
  - **Word Documents**: Extract text from paragraphs using `python-docx`.
- **Summarization**:
  - **Small Files**: Summarized using Groq LLM.
  - **Large Files**: Summarized using Google Gemini API for efficiency.
- **Usage**:
  - Upload a file using the UI's upload button.
  - The AI assistant will summarize and incorporate the file's content into the conversation.

## UI Features
- Real-time latency monitoring
- System status indicators
- Matrix-inspired background
- Custom scrollbars
- Responsive design
- Keyboard shortcuts (Enter to send)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
```






