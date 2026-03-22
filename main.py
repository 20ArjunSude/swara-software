from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.post("/generate")
def generate(req: PromptRequest):
    try:
        # 🔥 Strong classification prompt
        system_prompt = """
You are a strict command classifier.

Rules:
- Output ONLY one label from the list
- Do NOT explain
- If input does NOT clearly match → return "Fallback Intent"
- NEVER guess

Priority:
1. podcast → Podcast
2. music → Music
3. general play → Play / Resume

Allowed Labels:
Music
Joke
News
Podcast
Speech
Short Story
Map
Previous
Next
Play / Resume
Pause
Fallback Intent
"""

        response = client.chat.completions.create(
            model="llama3-70b-8192",  # ✅ upgraded model
            temperature=0,  # 🔥 important for consistency
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.prompt}
            ]
        )

        return {
            "answer": response.choices[0].message.content.strip()
        }

    except Exception as e:
        return {"error": str(e)}