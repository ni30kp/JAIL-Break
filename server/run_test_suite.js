const fs = require("fs");
const path = require("path");

// Import the RAGService from the separate file
const { RAGService } = require("./rag_service.js");

async function runTestSuite() {
  console.log("🚀 Starting Comprehensive Multi-Hop RAG Test Suite");
  console.log("=".repeat(60));

  // Load test questions
  const testData = JSON.parse(fs.readFileSync("./test_questions.json", "utf8"));
  const questions = testData.test_questions;

  console.log(
    `📋 Loaded ${questions.length} test questions across ${testData.metadata.categories.length} categories`
  );
  console.log("");

  // Initialize RAG service
  const ragService = new RAGService();
  await ragService.initialize();

  console.log("✅ RAG Service initialized and ready");
  console.log("");

  // Results storage
  const results = {
    test_run: {
      timestamp: new Date().toISOString(),
      total_questions: questions.length,
      categories: testData.metadata.categories,
    },
    results: [],
  };

  // Process each question
  for (let i = 0; i < questions.length; i++) {
    const question = questions[i];
    console.log(`\n🔍 Processing Question ${i + 1}/${questions.length}`);
    console.log(`📂 Category: ${question.category}`);
    console.log(`❓ Question: ${question.question}`);
    console.log("-".repeat(80));

    try {
      const startTime = Date.now();
      const response = await ragService.query(question.question);
      const endTime = Date.now();
      const processingTime = endTime - startTime;

      console.log(`⏱️  Processing time: ${processingTime}ms`);
      console.log(`🤖 Provider: ${response.provider}`);
      console.log(`📄 Sources: ${response.sources.length} documents`);
      console.log(`🔄 Hops: ${response.hops.length}`);

      // Store result
      results.results.push({
        id: question.id,
        category: question.category,
        question: question.question,
        answer: response.answer,
        provider: response.provider,
        sources_count: response.sources.length,
        hops: response.hops,
        processing_time_ms: processingTime,
        timestamp: new Date().toISOString(),
      });

      console.log(`✅ Question ${i + 1} completed successfully`);
    } catch (error) {
      console.error(`❌ Error processing question ${i + 1}:`, error.message);

      // Store error result
      results.results.push({
        id: question.id,
        category: question.category,
        question: question.question,
        error: error.message,
        timestamp: new Date().toISOString(),
      });
    }

    // Add a small delay between questions to avoid overwhelming the system
    if (i < questions.length - 1) {
      console.log("⏳ Waiting 2 seconds before next question...");
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  }

  // Generate summary statistics
  const successfulResults = results.results.filter((r) => !r.error);
  const failedResults = results.results.filter((r) => r.error);

  console.log("\n" + "=".repeat(60));
  console.log("📊 TEST SUITE SUMMARY");
  console.log("=".repeat(60));
  console.log(`✅ Successful: ${successfulResults.length}/${questions.length}`);
  console.log(`❌ Failed: ${failedResults.length}/${questions.length}`);
  console.log(
    `📈 Success Rate: ${(
      (successfulResults.length / questions.length) *
      100
    ).toFixed(1)}%`
  );

  if (successfulResults.length > 0) {
    const avgProcessingTime =
      successfulResults.reduce((sum, r) => sum + r.processing_time_ms, 0) /
      successfulResults.length;
    console.log(
      `⏱️  Average processing time: ${avgProcessingTime.toFixed(0)}ms`
    );

    const providerCounts = {};
    successfulResults.forEach((r) => {
      providerCounts[r.provider] = (providerCounts[r.provider] || 0) + 1;
    });
    console.log("🤖 Provider usage:", providerCounts);

    const avgSources =
      successfulResults.reduce((sum, r) => sum + r.sources_count, 0) /
      successfulResults.length;
    console.log(`📄 Average sources per answer: ${avgSources.toFixed(1)}`);
  }

  // Save results to file
  const resultsFile = `test_results_${new Date()
    .toISOString()
    .replace(/[:.]/g, "-")}.json`;
  fs.writeFileSync(resultsFile, JSON.stringify(results, null, 2));
  console.log(`\n💾 Results saved to: ${resultsFile}`);

  // Generate category breakdown
  console.log("\n📂 RESULTS BY CATEGORY:");
  console.log("-".repeat(40));

  const categoryStats = {};
  testData.metadata.categories.forEach((category) => {
    const categoryResults = results.results.filter(
      (r) => r.category === category
    );
    const categorySuccess = categoryResults.filter((r) => !r.error).length;
    const categoryTotal = categoryResults.length;

    categoryStats[category] = {
      total: categoryTotal,
      successful: categorySuccess,
      success_rate:
        categoryTotal > 0
          ? ((categorySuccess / categoryTotal) * 100).toFixed(1)
          : "0.0",
    };

    console.log(
      `${category}: ${categorySuccess}/${categoryTotal} (${categoryStats[category].success_rate}%)`
    );
  });

  console.log("\n🎉 Test suite completed!");

  return results;
}

// Run the test suite
if (require.main === module) {
  runTestSuite()
    .then(() => {
      console.log("\n✅ Test suite finished successfully");
      process.exit(0);
    })
    .catch((error) => {
      console.error("\n❌ Test suite failed:", error);
      process.exit(1);
    });
}

module.exports = { runTestSuite };
