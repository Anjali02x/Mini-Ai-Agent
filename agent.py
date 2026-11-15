!pip install -q requests ddgs

import requests, time, io
import pandas as pd
from ddgs import DDGS
from google.colab import userdata

API = "https://agents-course-unit4-scoring.hf.space"
KEY = userdata.get("Groq_Api_Key")
H = {"Authorization": f"Bearer {KEY}"}

live = [m["id"] for m in requests.get("https://api.groq.com/openai/v1/models", headers=H, timeout=30).json()["data"]]
prefs = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
MODELS = [m for m in prefs if m in live] or [m for m in live if "whisper" not in m and "guard" not in m][:2]
print("Using:", MODELS)

BAD = ("i'm sorry", "i’m sorry", "i couldn", "i can't", "i need the", "could you please")

def search(text):
    try:
        res = DDGS().text(text[:300], max_results=5)
        return "\n".join(f"- {r['title']}: {r['body']}" for r in res)
    except Exception:
        return ""

def file_text(q):
    if not q.get("file_name"):
        return ""
    f = requests.get(f"{API}/files/{q['task_id']}", timeout=60)
    if not f.ok:
        return ""
    name = q["file_name"].lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(f.content)).to_string()[:8000]
        if name.endswith((".csv", ".txt", ".py", ".json")):
            return f.content.decode("utf-8", "ignore")[:8000]
    except Exception:
        pass
    return ""

def ask(prompt):
    for m in MODELS:
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=H,
                json={"model": m, "messages": [{"role": "user", "content": prompt}]}, timeout=90)
            if r.ok:
                return r.json()["choices"][0]["message"]["content"].strip()
            print("  error:", r.status_code, r.text[:120])
        except Exception as e:
            print("  error:", str(e)[:120])
    return ""

def answer(q):
    extra = file_text(q)
    prompt = (f"Web search results:\n{search(q['question'])}\n\n"
              + (f"Attached file contents:\n{extra}\n\n" if extra else "")
              + f"Question: {q['question']}\nReply with ONLY the final answer, no explanation.")
    a = ""
    for attempt in range(3):
        a = ask(prompt)
        if a and not a.lower().startswith(BAD):
            return a
        time.sleep(20 * (attempt + 1))
    return a

questions = requests.get(f"{API}/questions", timeout=60).json()
answers = []
for q in questions:
    a = answer(q)
    print(q["task_id"][:8], "→", a)
    answers.append({"task_id": q["task_id"], "submitted_answer": a})
    time.sleep(8)
