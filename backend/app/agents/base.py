import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    json_mode: bool = False,
) -> str:
    """Single LLM call. Returns the assistant's text response."""
    kwargs = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = _client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    """LLM call expecting JSON output. Returns parsed dict."""
    text = call_llm(system_prompt, user_prompt, temperature=0.1, json_mode=True)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Llama occasionally returns malformed JSON — return a safe fallback
        return {"error": "invalid_json", "raw": text}