const { ChatOpenAI, OpenAIEmbeddings } = require("@langchain/openai");
const { Ollama, OllamaEmbeddings } = require("@langchain/ollama");
const { HNSWLib } = require("@langchain/community/vectorstores/hnswlib");
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const { EnsembleRetriever } = require("langchain/retrievers/ensemble");
const { PromptTemplate } = require("@langchain/core/prompts");
const { LLMChain } = require("@langchain/core/chains");
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

    console.log("Initializing RAG Service...");
    await this.initializeLLM();
    await this.initializeVectorStores();
    await this.setupChain();
    this.isInitialized = true;
    console.log("RAGService is ready for processing");
  }

  initializeLLM() {
    const openaiApiKey = process.env.OPENAI_API_KEY;
    const ollamaBaseUrl =
      process.env.OLLAMA_BASE_URL || "http://localhost:11434";

    if (openaiApiKey) {
      console.log("Using openai as primary LLM provider.");
      this.llm = new ChatOpenAI({
        openAIApiKey: openaiApiKey,
        modelName: "gpt-4o",
        temperature: 0.1,
        maxTokens: 2000,
      });
      console.log("OpenAI Model: gpt-4o");
    } else {
      console.log(
        "OpenAI API key not found, using Ollama as primary LLM provider."
      );
      this.llm = new Ollama({
        baseUrl: ollamaBaseUrl,
        model: "mistral:latest",
        temperature: 0.1,
      });
      console.log("Ollama Model: mistral:latest");
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
    const documentsPath = "./db_documents_mistral_latest";
    const transcriptionsPath = "./db_transcriptions_mistral_latest";

    try {
      // Load or create document vector store
      if (fs.existsSync(documentsPath)) {
        console.log(`Loading existing vector store from ${documentsPath}`);
        this.vectorStores.documents = await HNSWLib.load(
          documentsPath,
          this.embeddings
        );
      }

      // Load or create transcription vector store
      if (fs.existsSync(transcriptionsPath)) {
        console.log(`Loading existing vector store from ${transcriptionsPath}`);
        this.vectorStores.transcriptions = await HNSWLib.load(
          transcriptionsPath,
          this.embeddings
        );
      }

      console.log("HNSWLib vector stores initialized/loaded");
    } catch (error) {
      console.error("Error initializing vector stores:", error);
      throw error;
    }
  }

  async setupChain() {
    try {
      this.retrievalChain = await this.createChain(this.llm);
      console.log("Retrieval chain successfully set up.");
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
    console.log(`Invoking multi-hop RAG chain for query: ${question}`);

    try {
      const result = await this.multiHopQuery(question);
      console.log(
        `Multi-hop query completed successfully using ${result.provider}`
      );

      return {
        answer: result.answer,
        sources: result.sources || [],
        provider: result.provider,
        hops: result.hops || [],
      };
    } catch (error) {
      console.error("Error in multi-hop RAG query:", error);
      throw error;
    }
  }

  async multiHopQuery(question) {
    console.log("=== Starting Multi-Hop RAG Process ===");

    // First hop: Initial retrieval from both documents and transcripts (reduced to k=15 to avoid context length issues)
    console.log("🔄 HOP 1: Initial retrieval");
    const initialResults = await this.performRetrieval(question, 15);
    console.log(`Retrieved ${initialResults.length} initial documents`);

    // Generate follow-up questions based on initial results (increased to 8 questions)
    console.log("🔄 Generating follow-up questions");
    const followUpQuestions = await this.generateFollowUpQuestions(
      question,
      initialResults
    );
    console.log(
      `Generated ${followUpQuestions.length} follow-up questions:`,
      followUpQuestions
    );

    // Second hop: Retrieve additional information using follow-up questions (increased to k=8 per question)
    console.log("🔄 HOP 2: Follow-up retrieval");
    const additionalResults = [];
    for (const followUp of followUpQuestions) {
      const results = await this.performRetrieval(followUp, 8);
      additionalResults.push(...results);
      console.log(`Retrieved ${results.length} documents for: "${followUp}"`);
    }

    // Combine all results and remove duplicates
    const allResults = this.deduplicateResults([
      ...initialResults,
      ...additionalResults,
    ]);
    console.log(`Total unique documents after multi-hop: ${allResults.length}`);

    // Generate final answer using all retrieved information
    console.log("🔄 Generating final answer");
    const finalAnswer = await this.generateFinalAnswer(question, allResults);

    console.log("=== Multi-Hop RAG Process Complete ===");

    return {
      answer: finalAnswer.answer,
      sources: allResults.map((doc) => doc.metadata),
      provider: finalAnswer.provider,
      hops: [
        { hop: 1, query: question, results: initialResults.length },
        {
          hop: 2,
          queries: followUpQuestions,
          results: additionalResults.length,
        },
      ],
    };
  }

  async performRetrieval(query, k = 6) {
    try {
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
        console.log(`Ensemble retriever returned ${results.length} documents`);
      } else {
        // Use single retriever
        results = await validRetrievers[0].getRelevantDocuments(query);
        console.log(`Single retriever returned ${results.length} documents`);
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

    console.log(
      `Truncated ${documents.length} documents to ${truncatedDocs.length} documents (${totalChars} chars)`
    );
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

1. **EXPLICIT POLICY NUMBERS**: Look for any policy numbers mentioned (CD-150, CD-160, CS-043, CD-100, etc.) and ask for more details about them.

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
You are an expert assistant helping Community Corrections staff apply Colorado Community Corrections Standards, Program Guidelines, and client transcripts.

IMPORTANT: Follow this exact structure when answering:

1. **FIRST**: Explicitly identify the correct Standard(s) or Guideline(s) that apply to this situation. Look for specific policy numbers mentioned in the context (CD-150, CD-160, CS-043, CD-100, etc.) and cite them properly.

2. **SECOND**: Bring in relevant client transcript information to provide specific context about the client's situation, if applicable.

3. **THIRD**: If applicable, mention how this situation impacts incentives, eligibility, supervision decisions, or other relevant consequences.

4. **FOURTH**: Only if the situation warrants it, mention the grievance process or other appropriate next steps.

Adapt your response to the specific topic:
- For financial/fee questions: Focus on CD-150, CD-160, financial obligations, incentives
- For treatment/program questions: Focus on CD-100, treatment standards, program requirements
- For grievance questions: Focus on grievance procedures, appeal rights, timelines
- For supervision questions: Focus on supervision standards, monitoring requirements
- For any other topic: Focus on relevant policies, procedures, and consequences

Question: "${question}"

Context Information:
${contextText}

Provide a comprehensive answer that directly addresses the question using the information above. Be specific about policy numbers, procedures, and consequences.`;

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

    // Try primary provider (OpenAI if available, otherwise Ollama)
    try {
      if (openaiApiKey) {
        const openaiLLM = new ChatOpenAI({
          openAIApiKey: openaiApiKey,
          modelName: "gpt-4o",
          temperature: 0.1,
          maxTokens: 2000,
        });
        const chain = await this.createChain(openaiLLM);
        const result = await chain.invoke(input);
        return { result, provider: "openai" };
      } else {
        const ollamaLLM = new Ollama({
          baseUrl: ollamaBaseUrl,
          model: "mistral:latest",
          temperature: 0.1,
        });
        const chain = await this.createChain(ollamaLLM);
        const result = await chain.invoke(input);
        return { result, provider: "ollama" };
      }
    } catch (primaryError) {
      console.log(
        `Primary provider (${openaiApiKey ? "openai" : "ollama"}) failed: ${
          primaryError.message
        }`
      );

      // Try fallback provider
      try {
        console.log(`Falling back to: ${openaiApiKey ? "ollama" : "openai"}`);
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
        console.log(
          `Fallback provider (${
            openaiApiKey ? "ollama" : "openai"
          }) also failed: ${fallbackError.message}`
        );
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

    return new LLMChain({
      llm,
      prompt,
    });
  }
}

module.exports = { RAGService };
