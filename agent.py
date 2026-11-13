!pip install -q transformers torch accelerate requests

import requests, time, torch
from transformers import pipeline

API = "https://agents-course-unit4-scoring.hf.space"
USERNAME = "Anjali02x"
AGENT_CODE = "https://github.com/Anjali02x/Mini-AI-Agent"

print("Loading local AI model...")
# Using the model that worked earlier
pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", device_map="auto")

def answer(q):
    text = q["question"]
    if q.get("file_name"):
         text += " (Note: A file was attached, ignore it and answer the text.)"
    
    # We are forcing it to be extremely brief
    messages = [
        {"role": "system", "content": "You are a trivia expert. Reply with ONLY the exact answer (1 word max). Do not explain."},
        {"role": "user", "content": text}
    ]
    
    for _ in range(3):
        try:
            response = pipe(messages, max_new_tokens=10)
            return response[0]['generated_text'][-1]['content'].strip()
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)
    return ""

print("Fetching questions...")
questions = requests.get(f"{API}/questions", timeout=60).json()

answers = []
for q in questions:
    print(f"Answering task: {q['task_id'][:8]}...")
    a = answer(q)
    print(f"→ Answer: {a}")
    answers.append({"task_id": q["task_id"], "submitted_answer": a})
