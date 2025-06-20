from fastapi import FastAPI, HTTPException, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any, AsyncGenerator
from dotenv import load_dotenv
import os
import google.generativeai as genai
import json
import asyncio
import uuid

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise Exception("GOOGLE_API_KEY not found in .env")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System formatting instructions
SYSTEM_INSTRUCTIONS = """
You are FarMart AI. Respond ONLY using the following structured format:

[
  {
    "name": "Farmart.AIComponents.Text.V1",
    "props": {
      "text": "📊 **Market Rates**\\n\\n---\\n\\n### 🌽 Maize\\n• Bihar\\n  • Gulabbagh: ₹2180 / qtl"
    }
  },
  {
    "name": "Farmart.AIComponents.SuggestionChips.V1",
    "props": {
      "suggestions": [
        { "label": "🌽 Maize Buyers", "value": "Maize Buyers" },
        { "label": "📈 Sell Crops", "value": "Sell Crops" }
      ]
    }
  }
]

Return ONLY valid JSON array like above. Do not wrap in Markdown, text, or explanations.
"""

def create_response_payload(components: List[Dict[str, Any]], message_id: str = None, success: bool = True) -> Dict[str, Any]:
    """Create standardized response payload"""
    return {
        "_id": message_id or str(uuid.uuid4()),
        "content": {
            "role": "model",
            "structured_parts": [
                {
                    "components": components
                }
            ]
        },
        "metadata": {},
        "success": success
    }

def create_text_component(text: str) -> Dict[str, Any]:
    """Create a text component"""
    return {
        "name": "Farmart.AIComponents.Text.V1",
        "props": {
            "text": text
        }
    }

async def process_gemini_request(text: str) -> List[Dict[str, Any]]:
    """Process request with Gemini and return components"""
    try:
        gemini_response = model.generate_content([
            {"role": "model", "parts": [SYSTEM_INSTRUCTIONS]},
            {"role": "user", "parts": [text]}
        ])
        
        components: List[Dict[str, Any]] = json.loads(gemini_response.text)
        return components
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Gemini returned invalid JSON")

@app.post("/send_message")
async def send_message(
    text: str = Form(...),
    x_user_id: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None),
    x_chat_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    try:
        components = await process_gemini_request(text)
        return create_response_payload(components)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send_message_stream")
async def send_message_stream(
    text: str = Form(...),
    x_user_id: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None),
    x_chat_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    async def stream() -> AsyncGenerator[bytes, None]:
        message_id = str(uuid.uuid4())
        
        try:
            progress_steps = [
                "🔄 **Processing your request...**\n\n*Analyzing query parameters*",
                "🌾 **Fetching market data...**\n\n*Connecting to agricultural databases*",
                "📊 **Generating insights...**\n\n*Preparing personalized recommendations*"
            ]
            
            for i, step_text in enumerate(progress_steps, 1):
                progress_component = create_text_component(step_text)
                progress_payload = create_response_payload([progress_component], message_id)
                yield (json.dumps(progress_payload) + "\n").encode("utf-8")
                await asyncio.sleep(1)

            components = await process_gemini_request(text)
            
            final_payload = create_response_payload(components, message_id)
            yield (json.dumps(final_payload) + "\n").encode("utf-8")

        except Exception as e:
            error_component = create_text_component(f"❌ **Error occurred**\n\n*{str(e)}*")
            error_payload = create_response_payload([error_component], message_id, success=False)
            yield (json.dumps(error_payload) + "\n").encode("utf-8")

    return StreamingResponse(stream(), media_type="application/json")