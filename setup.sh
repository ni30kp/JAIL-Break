#!/bin/bash

echo "🚀 Setting up Multi-Hop RAG Application with Ollama..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js v16 or higher."
    exit 1
fi

echo "✅ Node.js found: $(node --version)"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama is not installed. Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✅ Ollama installed. Please restart your terminal or run: source ~/.bashrc"
    echo "   Then run this script again."
    exit 0
fi

echo "✅ Ollama found: $(ollama --version)"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠️  Ollama is not running. Starting Ollama..."
    echo "   Please run 'ollama serve' in a separate terminal and keep it running."
    echo "   Then run this script again."
    exit 0
fi

echo "✅ Ollama is running"

# Check if the model is available
if ! ollama list | grep -q "llama3.2:3b"; then
    echo "📥 Pulling llama3.2:3b model (this may take a while)..."
    ollama pull llama3.2:3b
fi

echo "✅ Model available"

# Install root dependencies
echo "📦 Installing root dependencies..."
npm install

# Install server dependencies
echo "📦 Installing server dependencies..."
cd server && npm install && cd ..

# Install client dependencies
echo "📦 Installing client dependencies..."
cd client && npm install && cd ..

# Create uploads directory
echo "📁 Creating uploads directory..."
mkdir -p uploads

# Create ChromaDB directory
echo "📁 Creating ChromaDB directory..."
mkdir -p chroma_db

# Check if .env file exists in server directory
if [ ! -f "server/.env" ]; then
    echo "⚠️  No .env file found in server directory."
    echo "📝 Creating server/.env file with default Ollama configuration:"
    echo ""
    echo "OLLAMA_BASE_URL=http://localhost:11434"
    echo "OLLAMA_MODEL=llama3.2:3b"
    echo "CHROMA_DB_PATH=./chroma_db"
    echo "PORT=3001"
    echo "NODE_ENV=development"
    echo ""
    
    cat > server/.env << EOF
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
CHROMA_DB_PATH=./chroma_db
PORT=3001
NODE_ENV=development
EOF
    
    echo "✅ .env file created"
else
    echo "✅ .env file found in server directory"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the application:"
echo "  npm run dev"
echo ""
echo "Or start separately:"
echo "  npm run server  # Terminal 1"
echo "  npm run client  # Terminal 2"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend: http://localhost:3001"
echo ""
echo "📚 Make sure Ollama is running: ollama serve" 