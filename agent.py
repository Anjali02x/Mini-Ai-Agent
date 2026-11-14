!pip install -q transformers torch accelerate requests

import requests, time, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

API = "https://agents-course-unit4-scoring.hf.space"
USERNAME = "Anjali02x"
AGENT_CODE = "https://github.com/Anjali02x/Mini-AI-Agent/blob/main/agent.py"

print("Loading Flan-T5 model directly...")
# We load the model manually to bypass the broken pipeline
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

def answer(q):
    text = q["question"]
    if q.get("file_name"):
         text += " (Note: A file was attached, ignore it and answer the text.)"
    
    prompt = f"Answer this question with ONLY the exact final answer (1-3 words).\nQuestion: {text}"
    
    try:
        # Tokenize and generate manually
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=20)
        # Decode the answer
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()
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
