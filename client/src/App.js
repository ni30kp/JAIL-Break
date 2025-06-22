import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Upload,
  MessageCircle,
  FileText,
  Mic,
  Send,
  Loader2,
  CheckCircle,
  AlertCircle,
  PlayCircle,
  X,
  Check,
} from "lucide-react";

// Helper to render check or cross icons
const renderResultIcon = (value) =>
  value ? (
    <Check className="w-5 h-5 text-green-500" />
  ) : (
    <X className="w-5 h-5 text-red-500" />
  );

// Evaluation results component
const EvaluationResults = ({ results, onClear }) => {
  if (!results || results.length === 0) return null;

  return (
    <div className="mt-8 bg-gray-50 rounded-lg p-6 shadow-inner">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-semibold text-gray-800">
          Evaluation Results
        </h3>
        <button onClick={onClear} className="text-gray-500 hover:text-gray-700">
          <X className="w-5 h-5" />
        </button>
      </div>

      {results.map((result, index) => (
        <div key={index} className="mb-6 pb-6 border-b last:border-b-0">
          <h4 className="font-bold text-gray-700 mb-2">
            Question {index + 1}: {result.question}
          </h4>
          <div className="bg-white rounded-lg p-4 mb-4 whitespace-pre-wrap text-sm text-gray-600">
            <p className="font-semibold text-gray-800 mb-2">
              Generated Answer:
            </p>
            {result.answer}
          </div>
          <div className="mb-2">
            <strong className="text-gray-800">
              Score: {result["Total Score"]} / 7
            </strong>
          </div>
          <table className="w-full text-sm text-left text-gray-500">
            <thead className="text-xs text-gray-700 uppercase bg-gray-100">
              <tr>
                <th scope="col" className="px-6 py-3">
                  Criteria
                </th>
                <th scope="col" className="px-6 py-3 text-center">
                  Result
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result["Criteria Breakdown"]).map(
                ([key, value]) => (
                  <tr key={key} className="bg-white border-b">
                    <th
                      scope="row"
                      className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap"
                    >
                      {key}
                    </th>
                    <td className="px-6 py-4 flex justify-center">
                      {renderResultIcon(value)}
                    </td>
                  </tr>
                )
              )}
            </tbody>
          </table>
          {result.Error && (
            <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-lg">
              <strong>Error:</strong> {result.Error}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

function App() {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [serverStatus, setServerStatus] = useState("checking");
  const [evaluating, setEvaluating] = useState(false);
  const [evalResults, setEvalResults] = useState(null);

  useEffect(() => {
    checkServerHealth();
  }, []);

  const checkServerHealth = async () => {
    try {
      const response = await axios.get("/api/health");
      setServerStatus(response.data.ollama_connected ? "ready" : "no-ollama");
    } catch (error) {
      setServerStatus("error");
    }
  };

  const handleFileUpload = async (event, type) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("type", type);
    files.forEach((file) => formData.append("files", file));

    try {
      const response = await axios.post("/api/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const newFiles = response.data.files.map((file) => ({
        id: file.documentId,
        name: file.filename,
        type: file.type,
        timestamp: new Date().toISOString(),
      }));

      setUploadedFiles((prev) => [...prev, ...newFiles]);
    } catch (error) {
      console.error("Upload error:", error);
      alert(
        "Upload failed: " + (error.response?.data?.error || "Server error")
      );
    } finally {
      setUploading(false);
    }
  };

  const handleSubmitQuestion = async (e) => {
    e.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setAnswer("");
    setSources([]);

    try {
      const response = await axios.post("/api/query", { question });
      setAnswer(response.data.answer);
      setSources(response.data.sources || []);
    } catch (error) {
      console.error("Query error:", error);
      setAnswer("Error: Unable to process your question. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleRunEvaluation = async () => {
    setEvaluating(true);
    setEvalResults(null);
    try {
      const response = await axios.get("/api/evaluate");
      setEvalResults(response.data);
    } catch (error) {
      console.error("Evaluation error:", error);
      alert(
        "Evaluation failed: " + (error.response?.data?.error || "Server error")
      );
    } finally {
      setEvaluating(false);
    }
  };

  const getStatusIcon = () => {
    switch (serverStatus) {
      case "ready":
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case "no-ollama":
        return <AlertCircle className="w-5 h-5 text-yellow-500" />;
      case "error":
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Loader2 className="w-5 h-5 text-gray-500 animate-spin" />;
    }
  };

  const getStatusText = () => {
    switch (serverStatus) {
      case "ready":
        return "Ready to use";
      case "no-ollama":
        return "Ollama not connected";
      case "error":
        return "Server connection error";
      default:
        return "Checking status...";
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Multi-Hop RAG Application
          </h1>
          <p className="text-gray-600 mb-4">
            Upload documents and transcriptions, then ask questions that
            cross-reference both
          </p>
          <div className="flex items-center justify-center gap-2 text-sm text-gray-500">
            {getStatusIcon()}
            <span>{getStatusText()}</span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
                <Upload className="w-6 h-6" />
                Upload Files
              </h2>
              {/* Document Upload */}
              <div className="mb-6">
                <h3 className="text-lg font-medium mb-3 flex items-center gap-2">
                  <FileText className="w-5 h-5 text-blue-500" />
                  Documents
                </h3>
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400">
                  <input
                    type="file"
                    multiple
                    accept=".txt,.pdf,.doc,.docx,.csv,.json"
                    onChange={(e) => handleFileUpload(e, "document")}
                    className="hidden"
                    id="document-upload"
                    disabled={uploading}
                  />
                  <label
                    htmlFor="document-upload"
                    className="cursor-pointer block"
                  >
                    <FileText className="w-12 h-12 text-gray-400 mx-auto mb-2" />
                    <p className="text-gray-600">
                      {uploading ? "Uploading..." : "Click to upload"}
                    </p>
                    <p className="text-sm text-gray-400 mt-1">
                      TXT, PDF, DOC, DOCX, CSV, JSON
                    </p>
                  </label>
                </div>
              </div>

              {/* Transcription Upload */}
              <div>
                <h3 className="text-lg font-medium mb-3 flex items-center gap-2">
                  <Mic className="w-5 h-5 text-green-500" />
                  Transcriptions
                </h3>
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-green-400">
                  <input
                    type="file"
                    multiple
                    accept=".txt,.pdf,.doc,.docx,.csv,.json"
                    onChange={(e) => handleFileUpload(e, "transcription")}
                    className="hidden"
                    id="transcription-upload"
                    disabled={uploading}
                  />
                  <label
                    htmlFor="transcription-upload"
                    className="cursor-pointer block"
                  >
                    <Mic className="w-12 h-12 text-gray-400 mx-auto mb-2" />
                    <p className="text-gray-600">
                      {uploading ? "Uploading..." : "Click to upload"}
                    </p>
                    <p className="text-sm text-gray-400 mt-1">
                      TXT, PDF, DOC, DOCX, CSV, JSON
                    </p>
                  </label>
                </div>
              </div>
            </div>

            {/* Uploaded Files & Eval Button */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-medium mb-3">Uploaded Files</h3>
              <div className="space-y-2 max-h-48 overflow-y-auto mb-4">
                {uploadedFiles.map((file) => (
                  <div
                    key={file.id}
                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                  >
                    <div className="flex items-center gap-2">
                      {file.type === "document" ? (
                        <FileText className="w-4 h-4 text-blue-500" />
                      ) : (
                        <Mic className="w-4 h-4 text-green-500" />
                      )}
                      <span className="text-sm font-medium">{file.name}</span>
                    </div>
                    <span className="text-xs text-gray-400">
                      {new Date(file.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                ))}
              </div>
              <button
                onClick={handleRunEvaluation}
                disabled={evaluating || serverStatus !== "ready"}
                className="w-full px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {evaluating ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <PlayCircle className="w-5 h-5" />
                )}
                Run Evaluation
              </button>
            </div>
          </div>

          {/* Chat & Results Section */}
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
                <MessageCircle className="w-6 h-6" />
                Ask Questions
              </h2>

              <form onSubmit={handleSubmitQuestion} className="mb-6">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Ask a question..."
                    className="flex-1 px-4 py-2 border rounded-lg"
                    disabled={loading || serverStatus !== "ready"}
                  />
                  <button
                    type="submit"
                    disabled={!question.trim() || loading}
                    className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </div>
              </form>

              {loading && (
                <div className="flex justify-center items-center p-4">
                  <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
                </div>
              )}

              {answer && (
                <div className="bg-gray-50 rounded-lg p-4">
                  <h3 className="font-medium mb-2 text-gray-900">Answer:</h3>
                  <div className="text-gray-700 whitespace-pre-wrap">
                    {answer}
                  </div>
                </div>
              )}

              {sources.length > 0 && (
                <div className="mt-4 bg-blue-50 rounded-lg p-4">
                  <h3 className="font-medium mb-2 text-gray-900">Sources:</h3>
                  <div className="space-y-1 max-h-40 overflow-y-auto">
                    {sources.map((source, index) => (
                      <div
                        key={index}
                        className="text-xs bg-white p-2 rounded border"
                      >
                        {source.original_name} (Chunk {source.chunk_index + 1})
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <EvaluationResults
              results={evalResults}
              onClear={() => setEvalResults(null)}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
