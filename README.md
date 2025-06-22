# Jail-break: Multi-Hop RAG System

A full-stack, single-page web application for multi-hop Retrieval-Augmented Generation (RAG) using local vector stores (HNSWLib), Ollama, and OpenAI. Supports advanced multi-hop question answering over both policy documents and client transcripts.

## 🚀 Quick Start

```bash
# Install dependencies
npm install
cd server && npm install

# Start the application
npm start
```

## 🏗️ Architecture

### Backend (Node.js)
- **LangChain.js** (modular packages)
- **Ollama** (local LLM)
- **OpenAI** (fallback LLM)
- **HNSWLib** (vector storage)
- **Express** (API server)

### Frontend (React)
- **React** with **Tailwind CSS**
- Real-time chat interface
- File upload capabilities

## 🦜🔗 LangChain Import Structure

**⚠️ IMPORTANT: This project uses the modern modular LangChain packages. Do NOT import from the root `langchain` package or subpaths.**

### Correct Import Pattern
```js
const { ChatOpenAI, OpenAIEmbeddings } = require("@langchain/openai");
const { Ollama, OllamaEmbeddings } = require("@langchain/ollama");
const { HNSWLib } = require("@langchain/community/vectorstores/hnswlib");
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const { EnsembleRetriever } = require("langchain/retrievers/ensemble");
const { PromptTemplate } = require("@langchain/core/prompts");
const { LLMChain } = require("langchain/chains");
```

### ❌ Common Import Errors
```js
// DON'T DO THIS:
const { OpenAI } = require("langchain/llms/openai");  // ❌
const { LLMChain } = require("@langchain/core/chains");  // ❌
const { PromptTemplate } = require("langchain/prompts");  // ❌
```

## 🧪 Multi-Hop RAG Test Suite

The system includes a comprehensive test suite with 30 high-quality questions covering all major policy areas:

### Running the Test Suite
1. **Stop the backend server** (the test suite runs directly on the vector store)
2. Navigate to the server directory:
   ```bash
   cd server
   ```
3. Run the test suite:
   ```bash
   node run_test_suite.js
   ```
4. Results will be saved as `test_results_YYYY-MM-DDTHH-MM-SS.json`

### Test Categories
- **Policy + Supervision** (5 questions)
- **Fees + Incentives** (5 questions)
- **Monitoring / Technology** (5 questions)
- **Grievance + Appeals** (5 questions)
- **Case Management + Individualization** (5 questions)
- **Principles of Effective Intervention** (5 questions)

### Example Test Questions
- "What supervision procedures apply if Robert's family business creates a conflict of interest with supervision staff?"
- "If Nathan's ankle monitor repeatedly fails to charge, what policies about maintenance and compliance apply?"
- "How should Nathan's financial hardship be considered when calculating eligibility for privilege passes?"

## 🛠️ Setup

### Prerequisites
- Node.js (v18+)
- Ollama (for local LLM)
- OpenAI API key (optional, for fallback)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Jail-break

# Install dependencies
npm install
cd server && npm install
```

### Environment Configuration
Copy `server/env.example` to `server/.env`:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:latest

# OpenAI Configuration (optional)
OPENAI_API_KEY=sk-your-key-here

# Server Configuration
PORT=3001
NODE_ENV=development
```

### Starting the Application
```bash
# Start both frontend and backend
npm start

# Or start individually
npm run server  # Backend only
npm run client  # Frontend only
```

## 🔧 Troubleshooting

### Import/Export Errors
If you encounter `ERR_PACKAGE_PATH_NOT_EXPORTED`:

1. **Check import paths**: Use only the modular `@langchain/*` packages
2. **Verify package versions**: Ensure all LangChain packages are compatible
3. **Clear node_modules**: Delete and reinstall if needed
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

### Port Conflicts
- Ensure only one backend instance is running
- Check if port 3001 is available
- Use `lsof -i :3001` to find conflicting processes

### Context Length Errors
The system automatically truncates context for LLM calls. If you still hit limits:
- Reduce the number of retrieved documents in `rag_service.js`
- Adjust truncation parameters in the `truncateDocuments` method

### Ollama Issues
- Ensure Ollama is running: `ollama serve`
- Check model availability: `ollama list`
- Verify base URL in environment variables

## 📊 Multi-Hop RAG Process

The system performs true multi-hop retrieval:

1. **Initial Retrieval**: 15 documents from both policy docs and transcripts
2. **Follow-up Generation**: 8 targeted questions based on initial results
3. **Secondary Retrieval**: 8 documents per follow-up question
4. **Deduplication**: Remove duplicate documents
5. **Final Synthesis**: Generate comprehensive answer with policy citations

### Performance Metrics
- **Retrieval**: 15 initial + 64 follow-up documents (79 total)
- **Processing**: ~2-5 seconds per question
- **Accuracy**: Policy-cited, comprehensive answers

## 📁 Project Structure

```
Jail-break/
├── client/                 # React frontend
│   ├── src/
│   ├── public/
│   └── package.json
├── server/                 # Node.js backend
│   ├── rag_service.js      # Multi-hop RAG implementation
│   ├── run_test_suite.js   # Test suite runner
│   ├── test_questions.json # 30 test questions
│   ├── db_documents_*/     # Vector stores
│   └── package.json
├── uploads/                # File upload directory
└── README.md
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the test suite
5. Submit a pull request

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- Built with [LangChain.js](https://js.langchain.com/)
- Powered by [Ollama](https://ollama.com/)
- Enhanced with [OpenAI](https://platform.openai.com/) 