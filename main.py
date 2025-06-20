from fastapi import FastAPI, HTTPException, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import os
import google.generativeai as genai
import json

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

@app.post("/send_message")
async def send_message(
    text: str = Form(...),
    x_user_id: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None),
    x_chat_id: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    try:
        # Compose Gemini chat-style input
        gemini_response = model.generate_content([
            {"role": "model", "parts": [SYSTEM_INSTRUCTIONS]},
            {"role": "user", "parts": [text]}
        ])

        # Try to parse Gemini's output as JSON
        try:
            components: List[Dict[str, Any]] = json.loads(gemini_response.text)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Gemini returned invalid JSON")

        # Wrap in ModelMessageContent format
        response_payload = {
    "_id": "123",
    "content": {
        "role": "model",
        "structured_parts": [
            {
                "components": components
            }
        ]
    },
    "metadata": {},
    "success": True
}


        return response_payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
