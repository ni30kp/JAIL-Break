# Jail-break: Multi-Hop RAG System

A full-stack, single-page web application for multi-hop Retrieval-Augmented Generation (RAG) using local vector stores (HNSWLib), Ollama, and OpenAI. Supports advanced multi-hop question answering over both policy documents and client transcripts.

![Architecture](https://img.shields.io/badge/Stack-React%20%2B%20Node.js-blue)
![AI](https://img.shields.io/badge/AI-LangChain%20%2B%20Ollama%20%2B%20OpenAI-green)
![Database](https://img.shields.io/badge/VectorDB-HNSWLib-orange)

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Clone and install dependencies
git clone <repository-url>
cd Jail-break
npm run install-all

# 2. Set up Ollama (if not already installed)
# Install Ollama from https://ollama.com
ollama pull mistral:latest

# 3. Start the application
npm run dev
```

**🌐 Access your application at:** `http://localhost:3000`

## 📋 Prerequisites

Before running this project, ensure you have:

- **Node.js** (v18 or higher) - [Download here](https://nodejs.org/)
- **Ollama** - [Download here](https://ollama.com/) (for local LLM)
- **OpenAI API Key** (optional, for cloud LLM) - [Get one here](https://platform.openai.com/)

### 🤖 Install Ollama & Model

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama service
ollama serve

# Pull the required model (in a new terminal)
ollama pull mistral:latest

# Verify installation
ollama list
```

## 🛠️ Step-by-Step Setup

### 1. **Clone & Install Dependencies**

```bash
git clone <repository-url>
cd Jail-break

# Install all dependencies (root, server, and client)
npm run install-all
```

### 2. **Environment Configuration**

The project comes with a pre-configured `.env` file in the `server/` directory. You can use it as-is or modify it:

```bash
# server/.env (already configured)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:latest
LLM_PROVIDER=openai
OPENAI_API_KEY=your-api-key-here
PORT=3001
NODE_ENV=development
```

**To use only local Ollama (no OpenAI):**
```bash
# Change this line in server/.env:
LLM_PROVIDER=ollama
# Comment out or remove: OPENAI_API_KEY=...
```

### 3. **Start the Application**

```bash
# Option 1: Start both server and client together (Recommended)
npm run dev

# Option 2: Start them separately
# Terminal 1: Start server
npm run server

# Terminal 2: Start client  
npm run client
```

### 4. **Verify Everything is Running**

You should see:
- ✅ **Server**: Running on `http://localhost:3001`
- ✅ **Client**: Running on `http://localhost:3000`
- ✅ **Ollama**: Service running on `http://localhost:11434`

## 🏗️ Architecture

### Backend (Node.js/Express)
- **LangChain.js** - AI orchestration framework
- **Ollama** - Local LLM inference
- **OpenAI** - Cloud LLM (fallback/alternative)
- **HNSWLib** - High-performance vector storage
- **Express** - RESTful API server

### Frontend (React)
- **React** with **Tailwind CSS**
- Real-time chat interface
- Drag-and-drop file uploads
- Responsive design

## 🔧 Common Issues & Solutions

### ❌ **"Proxy error: Could not proxy request"**
**Problem**: Server not running on port 3001

**Solution**:
```bash
# 1. Ensure Ollama is running
ollama serve

# 2. Start the server
cd server && npm run dev
```

### ❌ **"Something is already running on port 3000"**
**Problem**: Previous React instance still running

**Solution**:
```bash
# Kill the process using port 3000
lsof -ti:3000 | xargs kill -9

# Or use a different port
PORT=3002 npm start
```

### ❌ **"Error: listen tcp 127.0.0.1:11434: bind: address already in use"**
**Problem**: Ollama is already running

**Solution**: This is actually good! Ollama is running. Continue with starting your server.

### ❌ **Import/Export Errors**
**Problem**: Wrong LangChain imports

**Solution**: Use the modular packages:
```js
// ✅ Correct
const { ChatOpenAI } = require("@langchain/openai");
const { Ollama } = require("@langchain/ollama");

// ❌ Wrong
const { OpenAI } = require("langchain/llms/openai");
```

### ❌ **"Module not found" errors**
**Solution**:
```bash
# Clean install
rm -rf node_modules package-lock.json
rm -rf server/node_modules server/package-lock.json  
rm -rf client/node_modules client/package-lock.json

# Reinstall
npm run install-all
```

## 📊 How Multi-Hop RAG Works

1. **📥 Document Upload**: Upload PDFs, Word docs, or text files
2. **🔍 Initial Query**: User asks a complex question
3. **📋 First Retrieval**: System retrieves 15 most relevant documents
4. **🤔 Follow-up Generation**: AI generates 8 targeted sub-questions
5. **📋 Second Retrieval**: System retrieves 8 documents per sub-question (64 total)
6. **🔄 Deduplication**: Remove duplicate documents 
7. **✨ Final Synthesis**: Generate comprehensive answer with citations

**Result**: Policy-cited, multi-perspective answers from 79 total documents

## 🧪 Testing the System

### Upload Test Documents
1. Navigate to `http://localhost:3000`
2. Upload sample documents (PDFs, DOCX, TXT)
3. Wait for processing confirmation

### Try Example Questions
- "What supervision procedures apply if there's a conflict of interest?"
- "How should financial hardship be considered for privilege passes?"
- "What happens if monitoring equipment repeatedly fails?"

### Run the Test Suite
```bash
# Stop the server first
cd server

# Run comprehensive test suite
node run_test_suite.js

# View results
ls test_results_*.json
```

## 📁 Project Structure

```
Jail-break/
├── 📁 client/                  # React Frontend
│   ├── src/
│   │   ├── App.js             # Main application
│   │   ├── index.js           # Entry point
│   │   └── index.css          # Tailwind styles
│   ├── public/
│   └── package.json
├── 📁 server/                  # Node.js Backend
│   ├── index.js               # Express server
│   ├── rag_service.js         # Multi-hop RAG logic
│   ├── run_test_suite.js      # Test runner
│   ├── test_questions.json    # 30 test questions
│   ├── .env                   # Environment config
│   ├── 📁 db_documents*/      # Vector stores
│   ├── 📁 uploads/            # Uploaded files
│   └── package.json
├── package.json               # Root package config
├── setup.sh                  # Setup script
└── README.md                 # This file
```

## 🔧 Development Commands

```bash
# Install all dependencies
npm run install-all

# Start both server and client
npm run dev

# Start only server (port 3001)
npm run server

# Start only client (port 3000)  
npm run client

# Build for production
npm run build

# Run test suite
cd server && node run_test_suite.js
```

## 🌟 Features

- **📄 Multi-format Support**: PDF, DOCX, TXT file uploads
- **🧠 Intelligent Retrieval**: Multi-hop question answering
- **🏃‍♂️ Local & Cloud**: Ollama for privacy, OpenAI for power
- **📊 Real-time Processing**: Live chat interface
- **🔍 Smart Citations**: Policy-backed answers
- **🧪 Comprehensive Testing**: 30-question test suite
- **🎨 Modern UI**: Tailwind CSS responsive design

## 🚨 Security Notes

- **OpenAI API Key**: Keep your API key secure and never commit it to version control
- **Local Processing**: Ollama runs entirely locally for privacy-sensitive documents
- **File Uploads**: Files are stored locally in the `server/uploads/` directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Test with: `cd server && node run_test_suite.js`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[LangChain.js](https://js.langchain.com/)** - AI application framework
- **[Ollama](https://ollama.com/)** - Local LLM inference
- **[OpenAI](https://platform.openai.com/)** - Cloud LLM services
- **[HNSWLib](https://github.com/nmslib/hnswlib)** - High-performance vector search
- **[React](https://reactjs.org/)** - Frontend framework
- **[Tailwind CSS](https://tailwindcss.com/)** - Utility-first CSS

## 💬 Support

If you encounter any issues:

1. **Check this README** - Most common issues are covered above
2. **Run the test suite** - `cd server && node run_test_suite.js`
3. **Check the logs** - Look for error messages in the terminal
4. **Create an issue** - Include error messages and your environment details

---

**🎉 Happy coding! Your multi-hop RAG system is ready to process complex queries and provide intelligent, policy-backed answers.** 