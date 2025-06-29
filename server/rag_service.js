const { ChatOpenAI, OpenAIEmbeddings } = require("@langchain/openai");
const { Ollama, OllamaEmbeddings } = require("@langchain/ollama");
const { HNSWLib } = require("@langchain/community/vectorstores/hnswlib");
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const { EnsembleRetriever } = require("langchain/retrievers/ensemble");
const { PromptTemplate } = require("@langchain/core/prompts");
// const { LLMChain } = require("@langchain/core/chains"); // Removed - deprecated import
const fs = require("fs");
const path = require("path");

class RAGService {
  constructor() {
    this.llm = null;
    this.embeddings = null;
    this.vectorStores = {
      documents: null,
      transcriptions: null,
    };
    this.retrievalChain = null;
    this.isInitialized = false;
  }

  async initialize() {
    if (this.isInitialized) {
      return;
    }

    await this.initializeLLM();
    await this.initializeVectorStores();
    await this.setupChain();
    this.isInitialized = true;
    console.log("RAG Service initialized");
  }

  initializeLLM() {
    const openaiApiKey = process.env.OPENAI_API_KEY;
    const ollamaBaseUrl =
      process.env.OLLAMA_BASE_URL || "http://localhost:11434";

    if (openaiApiKey) {
      this.llm = new ChatOpenAI({
        openAIApiKey: openaiApiKey,
        modelName: "gpt-4o",
        temperature: 0.1,
        maxTokens: 2000,
      });
    } else {
      this.llm = new Ollama({
        baseUrl: ollamaBaseUrl,
        model: "mistral:latest",
        temperature: 0.1,
      });
    }

    // Initialize embeddings with the same model
    if (openaiApiKey) {
      this.embeddings = new OpenAIEmbeddings({
        openAIApiKey: openaiApiKey,
        modelName: "text-embedding-3-small",
      });
    } else {
      this.embeddings = new OllamaEmbeddings({
        baseUrl: ollamaBaseUrl,
        model: "mistral:latest",
      });
    }
  }

  async initializeVectorStores() {
    const documentsPath = path.join(__dirname, "db_documents");
    const transcriptionsPath = path.join(__dirname, "db_transcriptions");

    try {
      // Load documents vector store
      this.vectorStores.documents = await HNSWLib.load(
        documentsPath,
        this.embeddings
      );
    } catch (error) {
      // Vector store doesn't exist yet, it will be created when documents are added
    }

    try {
      // Load transcriptions vector store
      this.vectorStores.transcriptions = await HNSWLib.load(
        transcriptionsPath,
        this.embeddings
      );
    } catch (error) {
      // Vector store doesn't exist yet, it will be created when documents are added
    }
  }

  async setupChain() {
    try {
      this.retrievalChain = await this.createChain(this.llm);
    } catch (error) {
      console.error("Error setting up retrieval chain:", error);
      throw error;
    }
  }

  async loadOrCreateStore(dbPath) {
    if (fs.existsSync(dbPath)) {
      return await HNSWLib.load(dbPath, this.embeddings);
    } else {
      // Create empty store
      const emptyDocs = [];
      return await HNSWLib.fromDocuments(emptyDocs, this.embeddings);
    }
  }

  getRetriever() {
    const validRetrievers = [
      this.vectorStores.documents,
      this.vectorStores.transcriptions,
    ].filter(Boolean);

    if (validRetrievers.length === 0) {
      return null;
    }

    if (validRetrievers.length === 1) {
      return validRetrievers[0].asRetriever({
        k: 6,
        searchType: "similarity",
        scoreThreshold: 0.7,
      });
    }

    // Create ensemble retriever
    const retrievers = validRetrievers.map((store) =>
      store.asRetriever({
        k: 6,
        searchType: "similarity",
        scoreThreshold: 0.7,
      })
    );

    return new EnsembleRetriever({
      retrievers,
      weights: [0.6, 0.4], // Give more weight to documents
    });
  }

  async extractTextFromFile(filePath, fileType) {
    const text = fs.readFileSync(filePath, "utf8");
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    return await splitter.splitText(text);
  }

  async addDocuments(filePath, documentId, metadata = {}) {
    const fileType = path.extname(filePath).toLowerCase();
    const texts = await this.extractTextFromFile(filePath, fileType);

    const documents = texts.map((text, index) => ({
      pageContent: text,
      metadata: {
        ...metadata,
        documentId,
        chunkIndex: index,
        source: filePath,
      },
    }));

    // Add to appropriate vector store
    if (metadata.type === "transcription") {
      if (!this.vectorStores.transcriptions) {
        this.vectorStores.transcriptions = await HNSWLib.fromDocuments(
          documents,
          this.embeddings
        );
      } else {
        await this.vectorStores.transcriptions.addDocuments(documents);
      }
      await this.vectorStores.transcriptions.save(
        "./db_transcriptions_mistral_latest"
      );
    } else {
      if (!this.vectorStores.documents) {
        this.vectorStores.documents = await HNSWLib.fromDocuments(
          documents,
          this.embeddings
        );
      } else {
        await this.vectorStores.documents.addDocuments(documents);
      }
      await this.vectorStores.documents.save("./db_documents_mistral_latest");
    }

    return documents.length;
  }

  async query(question) {
    try {
      const result = await this.multiHopQuery(question);
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
    // First hop: Initial retrieval
    const initialResults = await this.performRetrieval(question, 15);

    // Generate follow-up questions based on initial results
    const followUpQuestions = await this.generateFollowUpQuestions(
      question,
      initialResults
    );

    // Second hop: Retrieve additional information using follow-up questions
    const additionalResults = [];
    for (const followUp of followUpQuestions) {
      const results = await this.performRetrieval(followUp, 8);
      additionalResults.push(...results);
    }

    // Combine all results and remove duplicates
    const allResults = this.deduplicateResults([
      ...initialResults,
      ...additionalResults,
    ]);

    // Generate final answer using all retrieved information
    const finalAnswer = await this.generateFinalAnswer(question, allResults);

    return finalAnswer;
  }

  async performRetrieval(query, k = 6) {
    try {
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
      } else {
        // Use single retriever
        results = await validRetrievers[0].getRelevantDocuments(query);
      }

      return results.slice(0, k);
    } catch (error) {
      console.error(`Error in retrieval for query "${query}":`, error);
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
- "What specific requirements or qualifications are mentioned in the documents?"
- "What are the exact procedures or steps outlined for this process?"
- "What specific consequences or outcomes are described in the documents?"

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
    return results.filter((doc) => {
      const key = `${doc.metadata.documentId}-${doc.metadata.chunkIndex}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }

  async invokeWithFallback(input) {
    const openaiApiKey = process.env.OPENAI_API_KEY;
    const ollamaBaseUrl =
      process.env.OLLAMA_BASE_URL || "http://localhost:11434";

    try {
      // Try primary provider first
      const result = await this.retrievalChain.invoke(input);
      return { result, provider: openaiApiKey ? "openai" : "ollama" };
    } catch (primaryError) {
      // Try fallback provider
      try {
        const fallbackLLM = openaiApiKey
          ? new Ollama({
              baseUrl: ollamaBaseUrl,
              model: "mistral:latest",
              temperature: 0.1,
            })
          : new ChatOpenAI({
              openAIApiKey: openaiApiKey,
              modelName: "gpt-4o",
              temperature: 0.1,
              maxTokens: 2000,
            });

        const chain = await this.createChain(fallbackLLM);
        const result = await chain.invoke(input);
        return { result, provider: openaiApiKey ? "ollama" : "openai" };
      } catch (fallbackError) {
        throw new Error(
          `Both providers failed. Primary: ${primaryError.message}, Fallback: ${fallbackError.message}`
        );
      }
    }
  }

  async createChain(llm) {
    const template = `You are a helpful AI assistant. Answer the following question based on the provided context.

Context: {input}

Answer:`;

    const prompt = new PromptTemplate({
      template,
      inputVariables: ["input"],
    });

    // Use the updated approach without LLMChain
    const { RunnableSequence } = require("@langchain/core/runnables");
    const { StringOutputParser } = require("@langchain/core/output_parsers");

    return RunnableSequence.from([prompt, llm, new StringOutputParser()]);
  }
}

module.exports = { RAGService };
