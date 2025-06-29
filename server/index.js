const express = require("express");
const cors = require("cors");
const multer = require("multer");
const path = require("path");
const fs = require("fs-extra");
const { v4: uuidv4 } = require("uuid");
const axios = require("axios");
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const { OllamaEmbeddings } = require("@langchain/ollama");
const { HNSWLib } = require("@langchain/community/vectorstores/hnswlib");
const pdf = require("pdf-parse");
const mammoth = require("mammoth");
const { Ollama } = require("@langchain/ollama");
const { PromptTemplate } = require("@langchain/core/prompts");
const {
  RunnableSequence,
  RunnablePassthrough,
} = require("@langchain/core/runnables");
const { StringOutputParser } = require("@langchain/core/output_parsers");
const { EnsembleRetriever } = require("langchain/retrievers/ensemble");
const {
  createStuffDocumentsChain,
} = require("langchain/chains/combine_documents");
const { createRetrievalChain } = require("langchain/chains/retrieval");
const { formatDocumentsAsString } = require("langchain/util/document");
const { ChatOpenAI } = require("@langchain/openai");
require("dotenv").config();

const { runEvaluation } = require("./eval");

const app = express();
const PORT = process.env.PORT || 3001;
const LLM_PROVIDER = process.env.LLM_PROVIDER || "openai";
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || "http://localhost:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "mistral:latest";
const DEBUG_MODE = process.env.DEBUG === "true";

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static("uploads"));

// Configuration
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const DB_DOCUMENTS_PATH = `./db_documents_${OLLAMA_MODEL.replace(
  /[^a-zA-Z0-9]/g,
  "_"
)}`;
const DB_TRANSCRIPTIONS_PATH = `./db_transcriptions_${OLLAMA_MODEL.replace(
  /[^a-zA-Z0-9]/g,
  "_"
)}`;

// Initialize embeddings (will be updated in RAGService constructor)
let embeddings = null;

// Text splitter for chunking documents
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

// File upload configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = "uploads";
    fs.ensureDirSync(uploadDir);
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueName = `${uuidv4()}-${file.originalname}`;
    cb(null, uniqueName);
  },
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    const allowedTypes = [
      "text/plain",
      "application/pdf",
      "application/msword",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "text/csv",
      "application/json",
    ];

    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error("Invalid file type"), false);
    }
  },
});

// Multi-hop RAG service
class RAGService {
  constructor() {
    this.vectorStores = {
      documents: null,
      transcriptions: null,
    };
    this.llm = null;
    this.chain = null;
    this.ready = false;
    this.initializeLLM();
    this.initializeVectorStores();
  }

  initializeLLM() {
    // Initialize both models for fallback capability
    this.openaiLLM = new ChatOpenAI({
      apiKey: OPENAI_API_KEY,
      modelName: "gpt-4o",
      temperature: 0.2,
    });

    this.ollamaLLM = new Ollama({
      baseUrl: OLLAMA_BASE_URL,
      model: OLLAMA_MODEL,
      temperature: 0.2,
      topP: 0.9,
    });

    // Set primary LLM based on provider preference
    this.llm = LLM_PROVIDER === "openai" ? this.openaiLLM : this.ollamaLLM;
    this.primaryProvider = LLM_PROVIDER;

    console.log(`Using ${LLM_PROVIDER} as primary LLM provider.`);

    // Initialize embeddings with the same model as primary LLM
    if (LLM_PROVIDER === "openai") {
      // For OpenAI, we'll use a compatible embedding model
      embeddings = new OllamaEmbeddings({
        model: OLLAMA_MODEL,
        baseUrl: OLLAMA_BASE_URL,
      });
    } else {
      embeddings = new OllamaEmbeddings({
        model: OLLAMA_MODEL,
        baseUrl: OLLAMA_BASE_URL,
      });
    }
  }

  async initializeVectorStores() {
    try {
      // Ensure DB directories exist
      await fs.ensureDir(DB_DOCUMENTS_PATH);
      await fs.ensureDir(DB_TRANSCRIPTIONS_PATH);

      // Load existing vector stores or create new ones
      this.vectorStores.documents = await this.loadOrCreateStore(
        DB_DOCUMENTS_PATH
      );
      this.vectorStores.transcriptions = await this.loadOrCreateStore(
        DB_TRANSCRIPTIONS_PATH
      );

      await this.setupChain();
      this.ready = true;
      console.log("RAG Service ready");
    } catch (error) {
      console.error("Error initializing vector stores:", error);
    }
  }

  async setupChain() {
    const retriever = this.getRetriever();
    if (!retriever) {
      return;
    }

    const promptTemplate = `
You are an intelligent assistant that analyzes documents and provides comprehensive, accurate answers based on the provided context.

For each question:
- Carefully analyze the user's query to understand what information they're seeking
- Review all relevant information from the provided documents
- Extract the most pertinent details that directly address the question
- Synthesize the information to provide a clear, well-structured answer
- **Always cite specific details, sections, or sources from the documents when making claims**
- If the documents contain insufficient information to fully answer the question, clearly state what information is available and what is missing
- Be objective and stick to the facts presented in the documents

Context from uploaded documents:
{context}

Question:
"{input}"

Answer:
`;

    const prompt = PromptTemplate.fromTemplate(promptTemplate);

    const combineDocsChain = await createStuffDocumentsChain({
      llm: this.llm,
      prompt,
      documentSeparator: "\n---\n",
    });

    this.chain = await createRetrievalChain({
      retriever,
      combineDocsChain,
    });
  }

  async loadOrCreateStore(dbPath) {
    try {
      // Check if the directory is empty or contains a store
      const files = await fs.readdir(dbPath);
      if (files.length > 0) {
        console.log(`Loading existing vector store from ${dbPath}`);
        return await HNSWLib.load(dbPath, embeddings);
      }
    } catch (e) {
      console.log(
        `No existing store found at ${dbPath}, a new one will be created upon first upload.`
      );
    }
    // Return a placeholder that can be replaced on first add
    return null;
  }

  getRetriever() {
    const docRetriever = this.vectorStores.documents
      ? this.vectorStores.documents.asRetriever({
          k: 20,
          searchType: "similarity",
          scoreThreshold: 0.5,
        })
      : null;
    const transRetriever = this.vectorStores.transcriptions
      ? this.vectorStores.transcriptions.asRetriever({
          k: 20,
          searchType: "similarity",
          scoreThreshold: 0.5,
        })
      : null;

    const validRetrievers = [docRetriever, transRetriever].filter(Boolean);
    if (validRetrievers.length === 0) {
      return null;
    }

    return validRetrievers.length > 1
      ? new EnsembleRetriever({
          retrievers: validRetrievers,
          weights: [0.6, 0.4],
        })
      : validRetrievers[0];
  }

  async extractTextFromFile(filePath, fileType) {
    try {
      const buffer = await fs.readFile(filePath);

      switch (fileType) {
        case "application/pdf":
          const pdfData = await pdf(buffer);
          return pdfData.text;

        case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        case "application/msword":
          const result = await mammoth.extractRawText({ buffer });
          return result.value;

        case "text/plain":
        case "text/csv":
        case "application/json":
        default:
          return buffer.toString("utf-8");
      }
    } catch (error) {
      console.error("Error extracting text:", error);
      throw error;
    }
  }

  async addDocuments(filePath, documentId, metadata = {}) {
    try {
      if (!this.ready) {
        throw new Error(
          "RAGService not ready. Please wait for initialization to complete."
        );
      }

      const fileContent = await this.extractTextFromFile(
        filePath,
        metadata.mimeType
      );

      if (!embeddings) {
        throw new Error("Embeddings not initialized");
      }

      const chunks = await textSplitter.splitText(fileContent);

      const storeType =
        metadata.type === "transcription" ? "transcriptions" : "documents";
      const dbPath =
        metadata.type === "transcription"
          ? DB_TRANSCRIPTIONS_PATH
          : DB_DOCUMENTS_PATH;

      const documentsWithMetadata = chunks.map((chunk, index) => ({
        pageContent: chunk,
        metadata: {
          ...metadata,
          chunk_index: index,
          total_chunks: chunks.length,
          document_id: documentId,
        },
      }));

      if (this.vectorStores[storeType]) {
        await this.vectorStores[storeType].addDocuments(documentsWithMetadata);
      } else {
        // Create a new store if it's the first upload
        this.vectorStores[storeType] = await HNSWLib.fromDocuments(
          documentsWithMetadata,
          embeddings
        );
      }

      await this.vectorStores[storeType].save(dbPath);
      await this.setupChain(); // Re-initialize the chain with the new data

      console.log(`Added document: ${documentId} (${chunks.length} chunks)`);
      return { success: true, chunks: chunks.length };
    } catch (error) {
      console.error("Error uploading document:", error);
      throw error;
    }
  }

  async query(question) {
    if (DEBUG_MODE) console.log(`Processing query: ${question}`);

    try {
      const result = await this.multiHopQuery(question);

      if (DEBUG_MODE) console.log(`Query completed using ${result.provider}`);

      return {
        answer: result.answer,
        sources: result.sources || [],
        provider: result.provider,
        hops: result.hops || [],
      };
    } catch (error) {
      console.error("Error in RAG query:", error);
      throw error;
    }
  }

  async multiHopQuery(question) {
    if (DEBUG_MODE) console.log("=== Starting Multi-Hop RAG Process ===");

    // First hop: Initial retrieval
    if (DEBUG_MODE) console.log("🔄 HOP 1: Initial retrieval");
    const initialResults = await this.performRetrieval(question, 15);
    if (DEBUG_MODE)
      console.log(`Retrieved ${initialResults.length} initial documents`);

    // Generate follow-up questions based on initial results
    if (DEBUG_MODE) console.log("🔄 Generating follow-up questions");
    const followUpQuestions = await this.generateFollowUpQuestions(
      question,
      initialResults
    );
    if (DEBUG_MODE)
      console.log(`Generated ${followUpQuestions.length} follow-up questions`);

    // Second hop: Retrieve additional information using follow-up questions
    if (DEBUG_MODE) console.log("🔄 HOP 2: Follow-up retrieval");
    const additionalResults = [];
    for (const followUp of followUpQuestions) {
      const results = await this.performRetrieval(followUp, 8);
      additionalResults.push(...results);
      if (DEBUG_MODE)
        console.log(`Retrieved ${results.length} documents for: "${followUp}"`);
    }

    // Combine all results and remove duplicates
    const allResults = this.deduplicateResults([
      ...initialResults,
      ...additionalResults,
    ]);
    if (DEBUG_MODE)
      console.log(
        `Total unique documents after multi-hop: ${allResults.length}`
      );

    // Generate final answer using all retrieved information
    if (DEBUG_MODE) console.log("🔄 Generating final answer");
    const finalAnswer = await this.generateFinalAnswer(question, allResults);

    if (DEBUG_MODE) console.log("=== Multi-Hop RAG Process Complete ===");

    return finalAnswer;
  }

  async performRetrieval(query, k = 6) {
    try {
      if (DEBUG_MODE)
        console.log(`Performing retrieval for query: "${query}" with k=${k}`);

      // Create retrievers with the specific k value for this query
      const docRetriever = this.vectorStores.documents
        ? this.vectorStores.documents.asRetriever({
            k: k,
            searchType: "similarity",
            scoreThreshold: 0.5,
          })
        : null;
      const transRetriever = this.vectorStores.transcriptions
        ? this.vectorStores.transcriptions.asRetriever({
            k: k,
            searchType: "similarity",
            scoreThreshold: 0.5,
          })
        : null;

      const validRetrievers = [docRetriever, transRetriever].filter(Boolean);
      if (validRetrievers.length === 0) {
        throw new Error("No retrievers available");
      }

      let results = [];
      if (validRetrievers.length > 1) {
        // Use ensemble retriever
        const ensembleRetriever = new EnsembleRetriever({
          retrievers: validRetrievers,
          weights: [0.6, 0.4],
        });
        results = await ensembleRetriever.getRelevantDocuments(query);
        if (DEBUG_MODE)
          console.log(
            `Ensemble retriever returned ${results.length} documents`
          );
      } else {
        // Use single retriever
        results = await validRetrievers[0].getRelevantDocuments(query);
        if (DEBUG_MODE)
          console.log(`Single retriever returned ${results.length} documents`);
      }

      return results.slice(0, k);
    } catch (error) {
      console.error(`Retrieval error: ${error.message}`);
      return [];
    }
  }

  // Helper method to truncate documents to prevent context length issues
  truncateDocuments(documents, maxCharsPerDoc = 2000, maxTotalChars = 8000) {
    let totalChars = 0;
    const truncatedDocs = [];

    for (const doc of documents) {
      const content = doc.pageContent;
      let truncatedContent = content;

      // Truncate individual document if too long
      if (content.length > maxCharsPerDoc) {
        truncatedContent =
          content.substring(0, maxCharsPerDoc) + "... [truncated]";
      }

      // Check if adding this document would exceed total limit
      if (totalChars + truncatedContent.length > maxTotalChars) {
        break;
      }

      truncatedDocs.push({
        ...doc,
        pageContent: truncatedContent,
      });
      totalChars += truncatedContent.length;
    }

    return truncatedDocs;
  }

  async generateFollowUpQuestions(originalQuestion, initialResults) {
    // Truncate documents to prevent context length issues
    const truncatedResults = this.truncateDocuments(initialResults, 1500, 6000);

    const contextText = truncatedResults
      .map((doc) => doc.pageContent)
      .join("\n\n");

    const followUpPrompt = `
Based on the original question and the initial retrieved documents, generate 8 specific follow-up questions that would help gather more comprehensive information.

Original Question: "${originalQuestion}"

Initial Retrieved Information:
${contextText}

IMPORTANT: Generate follow-up questions that are SPECIFIC and TARGETED to find missing information. Focus on:

1. **SPECIFIC REFERENCES**: Look for any specific references, numbers, or identifiers mentioned in the documents and ask for more details about them.

2. **SPECIFIC PROCEDURES**: Ask for detailed procedures related to the question topic.

3. **CLIENT-SPECIFIC CONTEXT**: If a client is mentioned, ask for their specific situation and history.

4. **RELATED POLICIES**: Ask for policies that might be related but not yet found.

5. **CONSEQUENCES AND NEXT STEPS**: Ask about what happens if the situation isn't addressed.

6. **IMPLEMENTATION DETAILS**: Ask about how policies are actually applied in practice.

7. **MISSING INFORMATION**: Ask about any gaps in the current information.

8. **SPECIFIC EXAMPLES**: Ask for concrete examples or case studies related to the situation.

Make your questions SPECIFIC and DIRECT. Examples:
- "What does policy CD-150 specifically say about financial obligations?"
- "What are the exact procedures for handling ankle monitor failures?"
- "What specific consequences are outlined for non-compliance with fee requirements?"

Format your response as a JSON array of strings.

Follow-up Questions:
`;

    try {
      const { result, provider } = await this.invokeWithFallback({
        input: followUpPrompt,
      });

      // Try to parse JSON response
      try {
        const questions = JSON.parse(result.answer);
        return Array.isArray(questions) ? questions.slice(0, 8) : [];
      } catch (parseError) {
        // If JSON parsing fails, try to extract questions from text
        const text = result.answer;
        const questions =
          text.match(/\d+\.\s*([^?\n]+[?])/g) ||
          text.match(/"([^"]+[?])"/g) ||
          text.match(/([^.\n]+[?])/g);

        return questions
          ? questions
              .slice(0, 8)
              .map((q) => q.replace(/^["\d\.\s]+|["\d\.\s]+$/g, ""))
          : [];
      }
    } catch (error) {
      console.error("Error generating follow-up questions:", error);
      return [];
    }
  }

  async generateFinalAnswer(question, allResults) {
    // Truncate documents to prevent context length issues
    const truncatedResults = this.truncateDocuments(allResults, 2000, 10000);

    const contextText = truncatedResults
      .map((doc) => doc.pageContent)
      .join("\n\n---\n\n");

    const finalPrompt = `
You are an intelligent assistant that analyzes documents and provides comprehensive answers based on the uploaded content.

IMPORTANT: Follow this structure when answering:

1. **FIRST**: Identify the most relevant information from the uploaded documents that directly addresses the question.

2. **SECOND**: If applicable, provide specific context, examples, or details found in the documents.

3. **THIRD**: Explain how the information relates to the question and any implications or conclusions.

4. **FOURTH**: If there are limitations or missing information, clearly state what additional details would be helpful.

Guidelines for your response:
- Base your answer strictly on the provided document content
- Cite specific details, sections, or information from the documents
- Be objective and factual
- If the documents don't contain sufficient information to fully answer the question, be honest about the limitations
- Structure your response clearly and logically

Question: "${question}"

Context Information from uploaded documents:
${contextText}

Provide a comprehensive answer that directly addresses the question using the information above. Be specific about the sources and details found in the documents.`;

    try {
      const { result, provider } = await this.invokeWithFallback({
        input: finalPrompt,
      });

      return {
        answer: result.answer,
        provider,
      };
    } catch (error) {
      console.error("Error generating final answer:", error);
      throw error;
    }
  }

  deduplicateResults(results) {
    const seen = new Set();
    const unique = [];

    for (const result of results) {
      const key = `${result.pageContent.substring(0, 100)}_${
        result.metadata.document_id
      }`;
      if (!seen.has(key)) {
        seen.add(key);
        unique.push(result);
      }
    }

    return unique;
  }

  async invokeWithFallback(input) {
    try {
      // Try primary provider first
      const result = await this.chain.invoke(input);
      return { result, provider: this.primaryProvider };
    } catch (error) {
      // Fallback to the other provider
      const fallbackProvider =
        this.primaryProvider === "openai" ? "ollama" : "openai";
      const fallbackLLM =
        this.primaryProvider === "openai" ? this.ollamaLLM : this.openaiLLM;

      try {
        // Recreate chain with fallback LLM
        const fallbackChain = await this.createChain(fallbackLLM);
        const result = await fallbackChain.invoke(input);
        return { result, provider: fallbackProvider };
      } catch (fallbackError) {
        throw new Error(
          `Both providers failed. Primary: ${error.message}, Fallback: ${fallbackError.message}`
        );
      }
    }
  }

  async createChain(llm) {
    // Create retrieval chain with the specified LLM
    const promptTemplate = `
You are an intelligent assistant that analyzes documents and provides comprehensive, accurate answers based on the provided context.

For each question:
- Carefully analyze the user's query to understand what information they're seeking
- Review all relevant information from the provided documents
- Extract the most pertinent details that directly address the question
- Synthesize the information to provide a clear, well-structured answer
- **Always cite specific details, sections, or sources from the documents when making claims**
- If the documents contain insufficient information to fully answer the question, clearly state what information is available and what is missing
- Be objective and stick to the facts presented in the documents

Context from uploaded documents:
{context}

Question:
"{input}"

Answer:
`;

    const prompt = PromptTemplate.fromTemplate(promptTemplate);
    const documentChain = await createStuffDocumentsChain({
      llm,
      prompt,
    });

    const retriever = this.getRetriever();
    if (!retriever) {
      throw new Error("No retrievers available");
    }

    return createRetrievalChain({
      combineDocsChain: documentChain,
      retriever,
    });
  }
}

const ragService = new RAGService();

// Routes
app.post("/api/upload", upload.array("files", 10), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: "No files uploaded" });
    }

    const { type } = req.body; // 'document' or 'transcription'
    const uploadedFiles = [];

    for (const file of req.files) {
      const filePath = file.path;
      const documentId = `${type}-${uuidv4()}`;

      const metadata = {
        type: type,
        original_name: file.originalname,
        size: file.size,
        mimeType: file.mimetype,
        upload_date: new Date().toISOString(),
      };

      // Upload to RAG system
      const result = await ragService.addDocuments(
        filePath,
        documentId,
        metadata
      );

      uploadedFiles.push({
        documentId,
        filename: file.originalname,
        type: type,
        chunks: result.chunks,
      });
    }

    res.json({
      success: true,
      files: uploadedFiles,
    });
  } catch (error) {
    console.error("Upload error:", error);
    res.status(500).json({ error: "Upload failed", details: error.message });
  }
});

app.post("/api/query", async (req, res) => {
  try {
    const { question } = req.body;

    if (!question) {
      return res.status(400).json({ error: "Question is required" });
    }

    // Perform RAG query
    const result = await ragService.query(question);

    res.json(result);
  } catch (error) {
    console.error("Query error:", error);
    res.status(500).json({ error: "Query failed", details: error.message });
  }
});

app.get("/api/evaluate", async (req, res) => {
  try {
    const results = await runEvaluation(ragService);
    res.json(results);
  } catch (error) {
    console.error("Evaluation error:", error);
    res
      .status(500)
      .json({ error: "Evaluation failed", details: error.message });
  }
});

app.get("/api/health", async (req, res) => {
  let ollamaConnected = false;
  try {
    const ollamaResponse = await axios.get(`${OLLAMA_BASE_URL}/api/tags`);
    ollamaConnected = ollamaResponse.status === 200;
  } catch (e) {
    // Ollama not connected
  }

  const storesInitialized =
    ragService.vectorStores.documents !== null ||
    ragService.vectorStores.transcriptions !== null;

  res.json({
    status: "healthy",
    ollama_connected: ollamaConnected,
    vector_stores_initialized: storesInitialized,
    llm_provider: LLM_PROVIDER,
    ollama_model: OLLAMA_MODEL,
  });
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error("Server error:", error);
  res.status(500).json({ error: "Internal server error" });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  if (LLM_PROVIDER === "ollama") {
    console.log(`Ollama URL: ${OLLAMA_BASE_URL}`);
    console.log(`Ollama Model: ${OLLAMA_MODEL}`);
  } else {
    console.log(`OpenAI Model: gpt-4o`);
  }
});
