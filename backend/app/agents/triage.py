from .base import call_llm_json

SYSTEM_PROMPT = """You are a query triage agent for a research paper Q&A system.

Classify the user's query into ONE of these categories:
- "factual_qa": A specific question about content in the documents (e.g., "What dataset was used?", "What are the limitations?")
- "summarize": A request to summarize, explain, or give an overview of a document or section
- "out_of_scope": Clearly unrelated to research papers (e.g., weather, sports, current events, capitals of countries)

Also decide if the system should attempt to answer at all.

Respond with strict JSON only:
{
  "category": "factual_qa" | "summarize" | "out_of_scope",
  "should_answer": true | false,
  "reasoning": "<one short sentence>"
}"""


def triage(query: str) -> dict:
    """Classify a query. Returns {category, should_answer, reasoning}."""
    result = call_llm_json(SYSTEM_PROMPT, f"Query: {query}")

    # Defensive fallback if JSON parsing failed
    if "error" in result:
        return {
            "category": "factual_qa",
            "should_answer": True,
            "reasoning": "triage failed, defaulting to factual_qa",
        }

    return {
        "category": result.get("category", "factual_qa"),
        "should_answer": result.get("should_answer", True),
        "reasoning": result.get("reasoning", ""),
    }