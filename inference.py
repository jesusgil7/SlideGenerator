#!/usr/bin/env python3
import sys
import re
import requests

# === Configuration ===
INVOKE_URL = "https://api.brev.dev/v1/chat/completions"
API_KEY    = "brev_api_-30Tr0kdRzKy3RTr4k9AtChl7TrL"
MODEL_NAME = "nvcf:nvidia/llama-3.1-nemotron-nano-8b-v1:dep-30TtAqxOULNBaELk6LSoLPKzNfV"

def run_inference(prompt: str) -> str:
    """
    Calls the model and returns exactly the <slide>…</slide> XML.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }
    payload = {
        "model":            MODEL_NAME,
        "messages":         [{"role": "user", "content": prompt}],
        "max_tokens":       512,
        "temperature":      0.0,
        "top_p":            1.0,
        "stream":           False,
        "presence_penalty": 0,
        "frequency_penalty":0,
    }
    resp = requests.post(INVOKE_URL, json=payload, headers=headers)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]

    # extract only the <slide>…</slide> portion
    match = re.search(r"<slide>.*?</slide>", raw, re.DOTALL)
    if not match:
        raise ValueError("Could not find <slide>…</slide> in model output.")
    return match.group(0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py \"<your slide prompt>\"", file=sys.stderr)
        sys.exit(1)

    # Build the guardrail schema instruction
    schema_instruction = (
        'Create a slide titled "Global Warming Causes" with four bullet points. '
        'Output ONLY valid XML in this exact schema, with NO additional tags or prose:\n'
        '<slide>\n'
        '  <title>…</title>\n'
        '  <bullet>…</bullet>\n'
        '  <bullet>…</bullet>\n'
        '  <bullet>…</bullet>\n'
        '  <bullet>…</bullet>\n'
        '</slide>'
    )

    # Use the user‑supplied prompt—but append the schema instruction
    user_prompt = " ".join(sys.argv[1:])
    final_prompt = f'{user_prompt}\n\n{schema_instruction}'

    # Run inference and write out slide.xml
    slide_xml = run_inference(final_prompt)
    with open("slide.xml", "w", encoding="utf-8") as f:
        f.write(slide_xml)
    print("✅ slide.xml generated")
