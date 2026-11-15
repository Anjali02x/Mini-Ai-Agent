!pip install -q requests ddgs

import requests, time, io, subprocess, tempfile
import pandas as pd
from ddgs import DDGS
from bs4 import BeautifulSoup
from google.colab import userdata

# Keep previous answers in case this run is worse
try:
    old_answers = answers
except NameError:
    old_answers = []

API = "https://agents-course-unit4-scoring.hf.space"
KEY = userdata.get("GROQ_API_KEY")
H = {"Authorization": f"Bearer {KEY}"}

live = [m["id"] for m in requests.get("https://api.groq.com/openai/v1/models", headers=H, timeout=30).json()["data"]]
prefs = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
MODELS = [m for m in prefs if m in live] or [m for m in live if "whisper" not in m and "guard" not in m][:2]
print("Using:", MODELS)

BAD = ("i'm sorry", "i’m sorry", "i couldn", "i can't", "i need the", "could you please")

RULES = """You answer exam questions. Think step by step, then end with one line:
FINAL ANSWER: <answer>
Format of <answer>: a number has no commas, no units, no % unless asked. Text is as short
as possible, no articles (a, an, the), no period at the end. A list is comma-separated
in the order asked. Never refuse: always give your best guess.
When a question asks for a scientific or technical classification, apply the strict definition."""

def page_text(url):
    try:
        h = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).text
        return BeautifulSoup(h, "html.parser").get_text(" ", strip=True)[:3000]
    except Exception:
        return ""

def search(text):
    try:
        res = list(DDGS().text(text[:300], max_results=5))
    except Exception:
        return ""
    out = [f"- {r['title']}: {r['body']}" for r in res]
    for r in res[:2]:
        out.append("PAGE: " + page_text(r["href"]))
    return "\n".join(out)

def file_text(q):
    if not q.get("file_name"):
        return ""
    f = requests.get(f"{API}/files/{q['task_id']}", timeout=60)
    if not f.ok:
        return ""
    name = q["file_name"].lower()
    try:
        if name.endswith((".mp3", ".wav", ".m4a")):
            t = requests.post("https://api.groq.com/openai/v1/audio/transcriptions",
                headers=H, files={"file": (q["file_name"], f.content)},
                data={"model": "whisper-large-v3"}, timeout=120)
            return t.json().get("text", "")[:8000] if t.ok else ""
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(f.content)).to_string()[:8000]
        if name.endswith(".py"):
            p = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
            p.write(f.content); p.close()
            run = subprocess.run(["python", p.name], capture_output=True, text=True, timeout=30)
            return ("Code:\n" + f.content.decode("utf-8", "ignore")[:4000]
                    + "\n\nOutput when run:\n" + (run.stdout or run.stderr)[:2000])
        if name.endswith((".csv", ".txt", ".json")):
            return f.content.decode("utf-8", "ignore")[:8000]
    except Exception:
        pass
    return ""

def ask(prompt, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    for m in MODELS:
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=H,
                json={"model": m, "messages": msgs, "temperature": 0}, timeout=90)
            if r.ok:
                return r.json()["choices"][0]["message"]["content"].strip()
            print("  error:", r.status_code, r.text[:120])
        except Exception as e:
            print("  error:", str(e)[:120])
    return ""

def answer(q):
    extra = file_text(q)
    rev = q["question"][::-1]
    query = ask("Write a short web search query (max 10 words) to help answer this question. "
                f"It may be written backwards; reversed version: {rev}\nOutput only the query.\n\n"
                f"{q['question']}") or q["question"]
    ctx = search(query) + "\n" + search(q["question"])
    prompt = (f"Web search results:\n{ctx}\n\n"
              + (f"Attached file contents:\n{extra}\n\n" if extra else "")
              + f"Question: {q['question']}\n"
              + f"(If the question above looks backwards, here it is reversed: {rev})")
    a = ""
    for attempt in range(3):
        raw = ask(prompt, RULES)
        a = raw.split("FINAL ANSWER:")[-1].strip().split("\n")[0].strip().rstrip(".") \
            if "FINAL ANSWER:" in raw else raw
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
