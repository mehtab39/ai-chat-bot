from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
import os
import time
import logging
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise Exception("GOOGLE_API_KEY not found in .env file")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

genai.configure(api_key=API_KEY)
try:
    model = genai.GenerativeModel(
        model_name="models/gemini-2.5-flash",
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            max_output_tokens=8192,
        )
    )
    logger.info("✅ Gemini model initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize Gemini model: {e}")
    model = None

app = FastAPI(title="Gemini + useChat Stream API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Attachment(BaseModel):
    url: str
    contentType: str
    name: Optional[str] = None

class ChatRequest(BaseModel):
    role: str
    content: str
    experimental_attachments: Optional[List[Attachment]] = []
    id: Optional[str] = None
    createdAt: Optional[str] = None
    parts: Optional[List[Dict[str, Any]]] = []
    options: Dict[str, Any] = {}
    systemInstruction: Optional[str] = None

class RegisterSessionRequest(BaseModel):
    systemInstruction: str

current_system_instruction: Optional[str] = None

def format_last_message_for_gemini(message: ChatRequest) -> List[Dict[str, Any]]:
    role = "user"
    parts = []

    if current_system_instruction:
        parts.append({"text": current_system_instruction})
    
    if message.content:
        parts.append({"text": message.content})

    for att in message.experimental_attachments or []:
            parts.append({"inline_data": {
                "mime_type": att.contentType,
                "data": f"<<{att.url}>>" 
            }})

    return [{"role": role, "parts": parts}]

async def generate_data_stream(message: ChatRequest) -> AsyncGenerator[str, None]:
    try:
        if not model:
            raise HTTPException(status_code=500, detail="Model not initialized")

        gemini_messages = format_last_message_for_gemini(message)
        if not gemini_messages:
            yield "0:I'm here to help!\n"
            return

        response = await asyncio.to_thread(
            model.generate_content,
            gemini_messages,
            stream=True
        )

        for chunk in response:
            if chunk.text:
                token = chunk.text.strip()
                if token:
                    yield f'0:{json.dumps(token)}\n'
                    await asyncio.sleep(0.01)

        yield f'd:{json.dumps({"finishReason": "stop"})}\n'

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f'0:{json.dumps("Sorry, something went wrong.")}\n'
        yield f'd:{json.dumps({"finishReason": "error"})}\n'

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    async def stream():
        async for chunk in generate_data_stream(request):
            yield chunk

    return StreamingResponse(
        stream(),
        media_type="text/plain",
        headers={
            "x-vercel-ai-data-stream": "v1",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.post("/register_session")
async def register_session(body: RegisterSessionRequest):
    global current_system_instruction
    current_system_instruction = body.systemInstruction.strip()
    logger.info("🔧 Updated global system instruction")
    return {"message": "System instruction registered"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)