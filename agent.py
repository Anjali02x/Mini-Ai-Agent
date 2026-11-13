!pip install -q llama-cpp-python requests huggingface_hub

import requests, time, os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

API = "https://agents-course-unit4-scoring.hf.space"
USERNAME = "Anjali02x"
AGENT_CODE = "https://github.com/Anjali02x/Mini-AI-Agent"

print("Downloading a smarter local AI model (Phi-3)... This takes 2 minutes.")
# Downloading a highly optimized model file (GGUF format)
model_path = hf_hub_download(
    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf", 
    filename="Phi-3-mini-4k-instruct-q4.gguf"
)

# Load the model
llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

def answer(q):
    text = q["question"]
    if q.get("file_name"):
         text += " (Note: A file was attached, ignore it and answer the text.)"
    
    prompt = f"<|user|>\nAnswer this question with ONLY the exact answer (1-3 words). Do not explain. If you don't know, guess.\nQuestion: {text}<|end|>\n<|assistant|>\n"
    
    for _ in range(3):
        try:
            response = llm(prompt, max_tokens=20, stop=["<|end|>", "\n"])
            return response['choices'][0]['text'].strip()
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
