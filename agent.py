!pip install -q transformers torch accelerate requests

import requests, time, torch
from transformers import pipeline

API = "https://agents-course-unit4-scoring.hf.space"
USERNAME = "Anjali02x"
AGENT_CODE = "https://github.com/Anjali02x/Mini-AI-Agent/blob/main/agent.py"

print("Loading fast local AI model...")
pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", device_map="auto")

def answer(q):
    text = q["question"]
    
    # A few-shot prompt to force the small model to be brief and accurate
    prompt = f"""You are a trivia expert. Answer the question with ONLY the exact final answer (1-3 words). Do not explain.
    
Question: What is the capital of France?
Answer: Paris

Question: Who wrote Romeo and Juliet?
Answer: William Shakespeare

Question: {text}
Answer:"""
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = pipe(messages, max_new_tokens=15)
        # Clean up the output
        ans = response[0]['generated_text'][-1]['content']
        ans = ans.replace("Answer:", "").strip()
        return ans
    except Exception as e:
        return ""

print("Fetching questions...")
questions = requests.get(f"{API}/questions", timeout=60).json()

answers = []
for q in questions:
    print(f"Answering task: {q['task_id'][:8]}...")
    a = answer(q)
    print(f"→ Answer: {a}")
    answers.append({"task_id": q["task_id"], "submitted_answer": a})

print("\nFinished! Run Cell 2 to submit.")
