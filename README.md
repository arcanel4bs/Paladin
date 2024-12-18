# Paladin - Open-Source Web Search Assistant

## The Legend
Paladin is your trusted AI search assistant, a noble assistant combining the **mighty power of Llama 3 (70B)** with the **ancient arts of knowledge-seeking** through LangGraph workflows. Like a faithful squire, it ventures forth through the vast realms of DuckDuckGo to gather wisdom and knowledge for your quests.

🎥 [Watch Demo Video](https://youtu.be/rVNVyk0s568)

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support%20the%20Project-yellow?style=for-the-badge&logo=buy-me-a-coffee)](https://buymeacoffee.com/arcanel4bs)

## Arsenal of Powers
- ⚔️ **Mighty LLM Core**: Powered by Llama 3 70B through Groq's API
- 🗺️ **Knowledge Questing**: Real-time exploration through DuckDuckGo
- 📜 **Scroll Analysis**: Interprets `.txt`, `.pdf`, `.doc`, and `.docx` manuscripts
- ⚡ **Strategic Workflow**: Multi-step processing for enhanced intelligence
- 🏰 **Noble Interface**: Medieval-inspired design with status indicators
- ⚜️ **Quest Metrics**: Built-in performance tracking and status monitoring

![Paladin Interface](/public/paladin-demo.png)

## Arsenal Components
- **Stronghold**: TailwindCSS, Animated Interface
- **Keep**: Flask, SQLite, LangChain, LangGraph
- **Armory**: Groq (Llama 3 70B) & Gemini (2.0 flash)
- **Scout**: DuckDuckGo Search API
- **Library**: Support for text, PDF, and Word manuscripts

## Joining the Quest

### Requirements
- Python 3.8+
- Groq API Key
- Google Gemini API Key (for lengthy manuscripts)
- DuckDuckGo Search capabilities

### Establishing Your Stronghold
1. Clone the repository:
```bash
git clone https://github.com/arcanel4bs/Paladin.git
cd paladin
```

2. Equip your arsenal:
```bash
pip install -r requirements.txt
```

3. Set up your credentials:
```bash
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
```

4. Prepare your stronghold:
```bash
flask db upgrade
```

5. Summon your Paladin:
```bash
python app.py
```

6. Enter the grand hall:
   Open your browser and navigate to `http://127.0.0.1:5000`

## Quest Workflow
Your Paladin follows a strategic approach to each quest:

1. **Quest Analysis**:
   - Interprets your inquiries and analyzes provided manuscripts
   - Determines if knowledge seeking is required

2. **Strategic Planning**:
   - Ventures forth through DuckDuckGo if additional knowledge is needed
   - Distills gathered information into valuable insights

3. **Noble Response**:
   - Crafts responses using:
     - Quest history
     - Manuscript analysis
     - Gathered knowledge

## Manuscript Processing

- **Supported Formats**: `.txt`, `.pdf`, `.doc`, `.docx`
- **Processing Methods**:
  - **Text Scrolls**: Direct interpretation
  - **PDF Tomes**: Extract content using `PyPDF2`
  - **Word Chronicles**: Extract content using `python-docx`
- **Strategic Analysis**:
  - **Brief Texts**: Processed by Groq LLM
  - **Lengthy Tomes**: Processed by Google Gemini for efficiency

## Interface Features
- Response timing
- Status indicators
- Themed background
- Custom scrollbars
- Responsive design
- Quick commands (Enter to send)

## Join the Guild
Contributions to this noble quest are welcome! Please submit your Pull Requests.

## Charter
This project operates under the MIT License - see the LICENSE file for details.
```






