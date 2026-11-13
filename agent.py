!pip install -q transformers torch accelerate requests

import requests, time, torch
from transformers import pipeline

API = "https://agents-course-unit4-scoring.hf.space"
USERNAME = "Anjali02x"
AGENT_CODE = "https://github.com/Anjali02x/Mini-AI-Agent"

print("Downloading local AI model (this takes 1-2 minutes)...")
pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", device_map="auto")

def answer(q):
    text = q["question"]
    if q.get("file_name"):
         text += " (Note: A file was attached, ignore it and answer the text.)"
    
    messages = [
        {"role": "system", "content": "Reply with ONLY the final answer."},
        {"role": "user", "content": text}
    ]
    
    for _ in range(3):
        try:
            response = pipe(messages, max_new_tokens=50)
            # Extract the generated text
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

print("Finished generating answers! You can now run Cell 2 to submit.")
