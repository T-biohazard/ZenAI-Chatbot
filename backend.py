from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from chatbot import run_chatbot
import uvicorn

app = FastAPI(title="ZenAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: List[List[str]] = []

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = run_chatbot(request.message, request.history)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    return {"message": "ZenAI is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  