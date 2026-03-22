import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/classify")
def classify(req: PromptRequest):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "user", "content": req.prompt}
            ],
            temperature=0,
            max_tokens=10
        )

        output = response.choices[0].message.content.strip()

        return {"label": output}

    except Exception:
        return {"label": "Fallback Intent"}