# Paladin - Open-Source Perplexity Alternative

![Paladin Interface](/public/paladin-demo.png)

## Overview
Paladin is a noble AI assistant that combines the divine power of Llama 3 (70B) with sacred knowledge seeking through LangGraph workflows. It features a radiant, holy chamber interface designed for enlightened human-AI interaction.

## Features
- ⚔️ **Divine LLM Integration**: Powered by Llama 3 70B through Groq's API
- 📜 **Sacred Knowledge Search**: Real-time DuckDuckGo search capabilities
- 📖 **Tome Analysis**: Upload `.txt`, `.pdf`, `.doc`, and `.docx` files for divine interpretation
- ✨ **Blessed Workflow**: Multi-step processing for enhanced wisdom
- 🏰 **Holy Chamber UI**: Celestial-inspired interface with divine status indicators
- 🌟 **Divine Metrics**: Built-in blessing tracking and sacred status monitoring

## Technical Stack
- **Frontend**: TailwindCSS, Divine CSS Animations
- **Backend**: Flask, LangChain, LangGraph
- **LLM**: Groq (Llama 3 70B)
- **Knowledge Seeking**: DuckDuckGo Search API
- **Scroll Processing**: Supports text, PDF, and Word documents

## Getting Started

### Prerequisites
- Python 3.8+
- Groq API Key
- Google Gemini API Key (for interpreting lengthy scrolls)
- DuckDuckGo Search capabilities
- Required Python packages: `PyPDF2`, `python-docx`, etc.

### Installation
1. Clone the sacred repository:
```bash
git clone https://github.com/arcanel4bs/paladin.git
cd paladin
```

2. Install the divine requirements:
```bash
pip install -r requirements.txt
```

3. Set up your sacred seals (environment variables):
```bash
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

4. Consecrate the database:
```bash
flask db upgrade
```

5. Invoke the divine interface:
```bash
python app.py
```

6. Access the holy chamber:
   Open your browser and navigate to `http://127.0.0.1:5000`

## Divine Workflow
The application employs a blessed multi-stage workflow:

1. **Prayer Processing**:
   - Interprets user supplications and analyzes uploaded scrolls
   - Determines if sacred knowledge seeking is required

2. **Divine Branching**:
   - Seeks additional wisdom through the DuckDuckGo oracle if needed
   - Distills gathered knowledge into divine insights

3. **Blessed Response**:
   - Channels responses using:
     - Prayer history
     - Scroll interpretations
     - Oracle's wisdom

## Scroll Processing

- **Sacred Formats**: `.txt`, `.pdf`, `.doc`, `.docx`
- **Divine Processing**:
  - **Text Scrolls**: Direct interpretation
  - **PDF Tomes**: Extract wisdom using `PyPDF2`
  - **Word Scrolls**: Extract knowledge using `python-docx`
- **Sacred Synthesis**:
  - **Brief Scrolls**: Interpreted by Groq LLM
  - **Ancient Tomes**: Interpreted by Google Gemini for divine efficiency

## Holy Chamber Features
- Divine response timing
- Blessed status indicators
- Celestial background
- Sacred scrollbars
- Responsive holy chamber
- Divine shortcuts (Enter to invoke)

## Contributing
Contributions to this sacred project are welcome! Please submit your divine Pull Requests.

## License
This holy project is blessed under the MIT License - see the LICENSE scroll for details.
```






