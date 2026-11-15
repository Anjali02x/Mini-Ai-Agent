!pip install -q requests ddgs

import requests, time
from ddgs import DDGS
from google.colab import userdata

API = "https://agents-course-unit4-scoring.hf.space"
KEY = userdata.get("Groq_Api_Key")
H = {"Authorization": f"Bearer {KEY}"}

# Ask Groq which models are live, then prefer the strongest text model
live = [m["id"] for m in requests.get("https://api.groq.com/openai/v1/models", headers=H, timeout=30).json()["data"]]
print("Live models:", live)
prefs = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
MODELS = [m for m in prefs if m in live] or [m for m in live if "whisper" not in m and "guard" not in m][:2]
print("Using:", MODELS)

def search(text):
    try:
        res = DDGS().text(text, max_results=5)
        return "\n".join(f"- {r['title']}: {r['body']}" for r in res)
    except Exception:
        return ""

def answer(q):
    prompt = (f"Web search results:\n{search(q['question'])}\n\nQuestion: {q['question']}\n"
              "Reply with ONLY the final answer, no explanation.")
    for attempt in range(4):
        for m in MODELS:
            try:
                r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=H,
                    json={"model": m, "messages": [{"role": "user", "content": prompt}]}, timeout=60)
                if r.ok:
                    return r.json()["choices"][0]["message"]["content"].strip()
                print("  error:", r.status_code, r.text[:150])
            except Exception as e:
                print("  error:", str(e)[:150])
        time.sleep(10 * (attempt + 1))
    return ""

questions = requests.get(f"{API}/questions", timeout=60).json()
answers = []
for q in questions:
    a = answer(q)
    print(q["task_id"][:8], "→", a)
    answers.append({"task_id": q["task_id"], "submitted_answer": a})
    time.sleep(3)
