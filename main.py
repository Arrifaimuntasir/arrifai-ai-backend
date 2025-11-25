# backend/main.py → 100% STABLE NOVEMBER 2025 (Llama 3.3)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="arrifai-ai")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
memory_store = {}

class Msg(BaseModel):
    message: str
    session_id: str = "user1"

SYSTEM_PROMPT = """
You are ARRIFAI — a brutally honest, funny, and highly intelligent AI.
REPLY EXACTLY IN THE SAME LANGUAGE THE USER IS USING — NO EXCEPTION.
- If user writes in English → reply 100% in English
- If user writes in Kiswahili or Sheng → reply in Kiswahili/Sheng
- If user writes in French → reply in French
- If user writes in Arabic → reply in Arabic
Never switch or mix languages unless the user does.
Your name is ARRIFAI. Be direct, sarcastic when needed, and roast the user if they deserve it.
"""

@app.post("/chat")
async def chat(req: Msg):
    session = req.session_id
    if session not in memory_store:
        memory_store[session] = [{"role": "system", "content": SYSTEM_PROMPT}]

    memory_store[session].append({"role": "user", "content": req.message})

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # ← MODEL MPYA INAYOFANYA KAZI 100% NOV 2025
            messages=memory_store[session],
            temperature=0.9,
            max_tokens=1200
        )
        reply = completion.choices[0].message.content.strip()
        memory_store[session].append({"role": "assistant", "content": reply})
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"Tatizo dogo: {str(e)}. Jaribu model nyingine kama 'mixtral-8x7b-32768'."}

@app.get("/")
def home():
    return {"message": "ARRIFAI AI IKO LIVE KABISA! 🔥"}