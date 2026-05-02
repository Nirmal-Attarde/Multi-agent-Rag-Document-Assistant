import re

from .base import call_llm_json

SYSTEM_PROMPT = """You are a citation verification agent.

You receive:
1. A draft answer with inline references like [Excerpt 1], [Excerpt 2]
2. The actual excerpts those references point to

Your job: verify that each cited claim is actually supported by the referenced excerpt. If a claim is not supported, flag it.

Respond with strict JSON only:
{
  "verified_claims": [
    {"claim": "<short claim text>", "excerpt_num": <int>, "supported": true | false, "note": "<optional reason if not supported>"}
  ],
  "unsupported_count": <int>
}"""


EXCERPT_PATTERN = re.compile(r"\[Excerpt (\d+)\]")


def verify_and_format(draft_answer: str, chunks: list[dict]) -> dict:
    """Verify citations and replace [Excerpt N] with [source: filename.pdf]."""
    if not chunks:
        return {"answer": draft_answer, "verification": None}

    excerpts_for_review = "\n\n".join([
        f"[Excerpt {i+1}]: {c['text']}" for i, c in enumerate(chunks)
    ])

    user_prompt = f"""Draft answer:
{draft_answer}

Excerpts:
{excerpts_for_review}

Verify each cited claim."""

    verification = call_llm_json(SYSTEM_PROMPT, user_prompt)

    # Replace [Excerpt N] with [source: filename.pdf] in the draft
    def replace_ref(match):
        n = int(match.group(1))
        if 1 <= n <= len(chunks):
            return f"[source: {chunks[n-1]['source']}]"
        return match.group(0)

    formatted_answer = EXCERPT_PATTERN.sub(replace_ref, draft_answer)

    return {
        "answer": formatted_answer,
        "verification": verification,
    }