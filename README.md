# LangSearch - AI-Powered Search Interface

![LangSearch Interface](/public/image1.png)

## Overview
LangSearch is a sophisticated AI chat interface that combines the power of Llama 3 (70B) with web search capabilities through LangGraph workflows. It features a cyberpunk-inspired laboratory interface designed for seamless human-AI interaction.

## Features
- 🤖 **Advanced LLM Integration**: Powered by Llama 3 70B through Groq's API
- 🔍 **Web Search Integration**: Real-time DuckDuckGo search capabilities
- 🔄 **LangGraph Workflow**: Multi-step processing for enhanced response quality
- 🎨 **Sci-fi Laboratory UI**: Cyberpunk-inspired interface with real-time status indicators
- ⚡ **Performance Metrics**: Built-in latency tracking and system status monitoring

## Technical Stack
- **Frontend**: TailwindCSS, Custom CSS Animations
- **Backend**: Flask, LangChain, LangGraph
- **LLM**: Groq (Llama 3 70B)
- **Search**: DuckDuckGo Search API

## Getting Started

### Prerequisites
- Python 3.8+
- Groq API Key
- DuckDuckGo Search capabilities

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/langsearch.git
cd langsearch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
GROQ_API_KEY=your_groq_api_key_here
```
```bash
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

4. Run the application:
```bash
python app.py
```

## How It Works
The application uses a two-stage LangGraph workflow:

1. **Input Processing**:
   - Analyzes user input
   - Performs web search when needed
   - Processes search results with LLM

2. **Response Generation**:
   - Generates contextual responses
   - Incorporates search results
   - Maintains conversation coherence

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






