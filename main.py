from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
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

client = Groq(api_key=api_key)

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.post("/generate")
def generate(req: PromptRequest):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": req.prompt}
            ],
            temperature=0.5,
            max_tokens=512,
        )

        return {"answer": response.choices[0].message.content or "Empty response"}

    except Exception as e:
        return {"error": str(e)}