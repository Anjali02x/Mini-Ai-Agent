!pip install -q google-genai requests

import requests, mimetypes, time
from google import genai
from google.genai import types
from google.colab import userdata

API = "https://agents-course-unit4-scoring.hf.space"
USERNAME = "Anjali02x"
AGENT_CODE = "https://github.com/Anjali02x/Mini-AI-Agent"

client = genai.Client(api_key=userdata.get("Gemini_Api_Key"))

config = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
    system_instruction="Reply with ONLY the final answer, no explanation. Use search when needed.",
)

def answer(q):
    parts = [q["question"]]
    if q.get("file_name"):
        f = requests.get(f"{API}/files/{q['task_id']}", timeout=60)
        if f.ok:
            mime = mimetypes.guess_type(q["file_name"])[0] or "text/plain"
            parts.append(types.Part.from_bytes(data=f.content, mime_type=mime))
    for _ in range(3):
        try:
            r = client.models.generate_content(model="gemini-1.5-flash", contents=parts, config=config)
            return (r.text or "").strip()
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(10)
    return ""

print("Fetching questions...")
questions = requests.get(f"{API}/questions", timeout=60).json()

answers = []
for q in questions:
    print(f"Answering task: {q['task_id'][:8]}...")
    a = answer(q)
    print(f"→ Answer: {a}")
    answers.append({"task_id": q["task_id"], "submitted_answer": a})

print("Submitting answers...")
res = requests.post(f"{API}/submit", json={
    "username": USERNAME, 
    "agent_code": AGENT_CODE, 
    "answers": answers
}, timeout=120)

print("Server Response:")
print(res.json())
