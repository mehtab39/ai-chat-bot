import os
import time
import asyncio
import logging
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not API_KEY:
    raise Exception("GOOGLE_API_KEY not found in .env file")

# Configure Gemini
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
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    id: Optional[str] = None
    role: str  # "user", "assistant", or "system"
    content: str
    parts: Optional[List[Dict[str, Any]]] = None


class ChatRequest(BaseModel):
    messages: List[Message]
    options: Dict[str, Any] = {}
    id: Optional[str] = None


def format_messages_for_gemini(messages: List[Message]) -> List[Dict[str, Any]]:
    gemini_messages = []
    for msg in messages:
        if msg.role == "assistant":
            role = "model"
        elif msg.role == "system":
            continue  # Ignore system messages here
        else:
            role = "user"
        gemini_messages.append({
            "role": role,
            "parts": [{"text": msg.content}]
        })
    return gemini_messages


async def generate_data_stream(messages: List[Message]) -> AsyncGenerator[str, None]:
    try:
        if not model:
            raise HTTPException(status_code=500, detail="Model not initialized")

        gemini_messages = format_messages_for_gemini(messages)
        if not gemini_messages:
            yield "0:I'm here to help!\n"
            return

        # Extract the last user message
        system_msg = next((m.content for m in messages if m.role == "system"), None)
        last_user_msg = gemini_messages[-1]["parts"][0]["text"]
        if system_msg:
            last_user_msg = f"System: {system_msg}\nUser: {last_user_msg}"

        # Start Gemini chat
        chat = model.start_chat(history=gemini_messages[:-1])

        # Get stream response (in a separate thread)
        response = await asyncio.to_thread(
            chat.send_message,
            last_user_msg,
            stream=True
        )

        total_tokens = 0
        for chunk in response:
            if chunk.text:
                token = chunk.text.strip()
                if token:
                    total_tokens += 1
                    yield f'0:{json.dumps(token)}\n'  # Wrap text in JSON string (e.g. "Hello")
                   # await asyncio.sleep(0.01)  # Optional: smooth streaming

        # Send metadata at end
        yield f'd:{json.dumps({"finishReason": "stop", "usage": {"promptTokens": total_tokens, "completionTokens": total_tokens}})}\n'

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f'0:{json.dumps("Sorry, something went wrong.")}\n'
        yield f'd:{json.dumps({"finishReason": "error"})}\n'


# ----- Chat Endpoint -----
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    async def stream():
        async for chunk in generate_data_stream(request.messages):
            yield chunk

    return StreamingResponse(
        stream(),
        media_type="text/plain",
        headers={
            "x-vercel-ai-data-stream": "v1",  # REQUIRED for useChat
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
