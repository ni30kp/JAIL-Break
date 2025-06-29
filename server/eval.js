const qa_dataset = [
  {
    question:
      "What process would handle a situation where Robert's son working in his business led to conflict with supervision requirements?",
    // We don't have a ground truth answer here, we just evaluate the generated one.
  },
  // Add more of the 30 questions here as they become available.
];

function evaluateAnswer(answer) {
  if (!answer || typeof answer !== "string") {
    return {
      "Total Score": 0,
      "Criteria Breakdown": {
        "CS-043": false,
        "CD-100 or Responsivity": false,
        "Primary staff handling": false,
        "Avoids hallucination": false,
        "Grievance fallback": false,
        "Cites sources": false,
        "No unrelated policy": false,
      },
      Error: "Invalid or empty answer provided.",
    };
  }

  const criteria = {
    "CS-043": /CS-043/.test(answer),
    "CD-100 or Responsivity": /CD-100|Responsivity/i.test(answer),
    "Primary staff handling":
      /staff/i.test(answer) &&
      (/document/i.test(answer) || /refer/i.test(answer)),
    "Avoids hallucination": ![
      /physical force/i,
      /performance improvement plan/i,
      /training plan/i,
      /CCC-020/i,
    ].some((regex) => regex.test(answer)),
    "Grievance fallback":
      /grievance/i.test(answer) || /Grievance Policy 148/.test(answer),
    "Cites sources": /CS-|CD-|Principle|Grievance Policy/i.test(answer),
    "No unrelated policy": ![
      /eligibility to work at facility/i,
      /advisory board as first step/i,
      /CCC-020/i,
    ].some((regex) => regex.test(answer)),
  };

  const score = Object.values(criteria).filter(Boolean).length;

  return {
    "Total Score": score,
    "Criteria Breakdown": criteria,
  };
}

async function runEvaluation(ragService) {
  const evaluationResults = [];

  for (const qa_pair of qa_dataset) {
    const { question } = qa_pair;
    try {
      const result = await ragService.query(question);
      const answer = result.answer;
      const evalResult = evaluateAnswer(answer);
      evaluationResults.push({
        question,
        answer,
        ...evalResult,
      });
    } catch (error) {
      console.error(`Error processing question: ${question}`, error);
      evaluationResults.push({
        question,
        answer: "Error during generation.",
        "Total Score": 0,
        "Criteria Breakdown": {},
        Error: error.message,
      });
    }
  }

  return evaluationResults;
}

module.exports = { runEvaluation, evaluateAnswer };
