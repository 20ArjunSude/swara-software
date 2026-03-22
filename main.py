from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
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

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-3.1-flash-lite")

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.post("/generate")
def generate(req: PromptRequest):
    try:
        response = model.generate_content(req.prompt)
        return {"answer": response.text or "Empty response"}
    except Exception as e:
        return {"error": str(e)}