"use client";

import { useState, useRef, useEffect } from "react";

type Source = {
  source: string;
  score: number;
  preview: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  trace?: any;
};

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/chat`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: input, top_k: 5 }),
        }
      );

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
        sources: data.sources,
        trace: data.trace,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage: Message = {
        role: "assistant",
        content: `Error: ${err instanceof Error ? err.message : "Something went wrong"}`,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <header className="bg-white border-b px-6 py-4">
        <h1 className="text-xl font-semibold text-gray-900">
          Multi-Agent RAG Document Assistant
        </h1>
        <p className="text-sm text-gray-500">
          Ask questions about your indexed research papers
        </p>
      </header>

      <main className="flex-1 overflow-y-auto px-6 py-6">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center text-gray-400 mt-20">
              <p className="text-sm">Start by asking a question about your documents.</p>
              <p className="text-xs mt-2">
                Try: &quot;What is the main contribution of the paper?&quot;
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-2xl rounded-lg px-4 py-3 ${
                  msg.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-white border border-gray-200 text-gray-900"
                }`}
              >
                <div className="whitespace-pre-wrap text-sm leading-relaxed">
                  {msg.content}
                </div>

                {msg.sources && msg.sources.length > 0 && (
                  <details className="mt-3 pt-3 border-t border-gray-200">
                    <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                      {msg.sources.length} source{msg.sources.length > 1 ? "s" : ""}
                    </summary>
                    <div className="mt-2 space-y-2">
                      {msg.sources.map((src, j) => (
                        <div
                          key={j}
                          className="text-xs bg-gray-50 border border-gray-100 rounded p-2"
                        >
                          <div className="flex justify-between items-center mb-1">
                            <span className="font-medium text-gray-700">
                              {src.source}
                            </span>
                            <span className="text-gray-400">
                              score: {src.score.toFixed(3)}
                            </span>
                          </div>
                          <div className="text-gray-600 italic">
                            {src.preview}...
                          </div>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
                {msg.trace && (
                  <details className="mt-2 pt-2">
                    <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                      agent trace
                    </summary>
                    <pre className="mt-2 text-xs bg-gray-50 border border-gray-100 rounded p-2 overflow-x-auto">
                      {JSON.stringify(msg.trace, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 rounded-lg px-4 py-3">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.15s" }}
                  />
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.3s" }}
                  />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer className="bg-white border-t px-6 py-4">
        <div className="max-w-3xl mx-auto flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents..."
            rows={1}
            disabled={isLoading}
            className="flex-1 resize-none rounded-lg border border-gray-300 px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </footer>
    </div>
  );
}