from fastapi import FastAPI, HTTPException, Form, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional, AsyncGenerator
from dotenv import load_dotenv
import os
import google.generativeai as genai
import asyncio
import logging
import json
import re
from contextlib import asynccontextmanager
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise Exception("GOOGLE_API_KEY not found in .env file")

# Configure Gemini
genai.configure(api_key=API_KEY)

# Global model instance with error handling
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
except Exception as e:
    logger.error(f"Failed to initialize Gemini model: {e}")
    raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting FarMart AI API server...")
    yield
    # Shutdown
    logger.info("Shutting down FarMart AI API server...")

app = FastAPI(
    title="FarMart AI API",
    description="AI-powered agricultural marketplace assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# System formatting instructions
SYSTEM_INSTRUCTIONS = """
You are FarMart AI. Stream your response using the following protocol:

Each message block MUST strictly follow this format:

type: <event-type>
data: <content>

Each block MUST be separated by TWO newlines (\n\n).
Do NOT wrap anything in Markdown, quotes, or explanations.
Each data block must contain raw content based on its type.

CRITICAL: For JSON data, ensure the ENTIRE JSON object is complete and valid. Never split JSON across multiple chunks.

Supported event types:

- `text`: For plain updates or human-readable responses.
- `progress`: For status messages like "fetching data" or "generating response".
- `json`: For structured UI components. Must be a COMPLETE and valid JSON object with `name` and `props`.
- `error`: For error messages.
- `done`: To indicate end of the stream.

Example:

type: progress
data: 🔀 Processing your request...

type: text
data: Here is the complete response text without breaking.

type: json
data: {"name": "Farmart.AIComponents.Text.V1", "props": {"text": "📊 Market Rates\\n\\n### 🌽 Maize\\n• Bihar\\n  • Gulabbagh: ₹2180 / qtl"}}

type: json
data: {"name": "Farmart.AIComponents.SuggestionChips.V1", "props": {"suggestions": [{"label": "🌽 Maize Buyers", "value": "Maize Buyers"}]}}

type: done
data: [DONE]

IMPORTANT: Never break JSON objects across multiple chunks. Always ensure JSON is complete and parseable.
Strictly follow this structure. Return ONLY the stream, no markdown, no bullet points, no commentary.
"""

class StreamBuffer:
    """Buffer to accumulate and process stream chunks"""
    
    def __init__(self):
        self.buffer = ""
        self.current_block = ""
        self.current_type = None
        self.current_data = ""
    
    def add_chunk(self, chunk_text: str) -> list:
        """Add chunk to buffer and return complete blocks"""
        self.buffer += chunk_text
        complete_blocks = []
        
        # Process complete blocks separated by double newlines
        while '\n\n' in self.buffer:
            block, self.buffer = self.buffer.split('\n\n', 1)
            if block.strip():
                processed_block = self._process_block(block.strip())
                if processed_block:
                    complete_blocks.append(processed_block)
        
        return complete_blocks
    
    def _process_block(self, block: str) -> Optional[str]:
        """Process a complete block and validate it"""
        lines = block.strip().split('\n')
        if len(lines) < 2:
            return None
        
        type_line = lines[0].strip()
        data_lines = lines[1:]
        
        if not type_line.startswith('type: '):
            return None
        
        block_type = type_line[6:].strip()
        
        # Join all data lines
        data_content = '\n'.join(line[6:] if line.startswith('data: ') else line for line in data_lines)
        
        # Validate JSON blocks
        if block_type == 'json':
            try:
                json.loads(data_content)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in block: {data_content}")
                return None
        
        # Reconstruct the block
        return f"type: {block_type}\ndata: {data_content}\n\n"
    
    def get_remaining(self) -> Optional[str]:
        """Get any remaining content in buffer"""
        if self.buffer.strip():
            return self._process_block(self.buffer.strip())
        return None

class StreamError(Exception):
    """Custom exception for streaming errors"""
    pass

def extract_and_fix_json_chunks(text: str) -> str:
    """Extract and fix broken JSON chunks from text"""
    # Look for JSON patterns
    json_pattern = r'\{"name":[^}]+(?:\{[^}]*\})*[^}]*\}'
    
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            # Try to parse the JSON
            json.loads(match)
        except json.JSONDecodeError:
            # Try to fix common issues
            fixed_match = match
            
            # Fix missing closing braces
            open_braces = fixed_match.count('{')
            close_braces = fixed_match.count('}')
            
            if open_braces > close_braces:
                fixed_match += '}' * (open_braces - close_braces)
            
            # Fix missing closing brackets
            open_brackets = fixed_match.count('[')
            close_brackets = fixed_match.count(']')
            
            if open_brackets > close_brackets:
                fixed_match += ']' * (open_brackets - close_brackets)
            
            # Fix missing quotes
            if not fixed_match.endswith('"}') and not fixed_match.endswith(']}'):
                if fixed_match.endswith('"'):
                    fixed_match += '}'
                elif not fixed_match.endswith('}'):
                    fixed_match += '"}'
            
            try:
                json.loads(fixed_match)
                text = text.replace(match, fixed_match)
                logger.info(f"Fixed JSON: {match} -> {fixed_match}")
            except json.JSONDecodeError:
                logger.warning(f"Could not fix JSON: {match}")
    
    return text

async def create_error_stream(error_message: str) -> AsyncGenerator[bytes, None]:
    """Create an error stream response"""
    yield f"type: error\ndata: {error_message}\n\n".encode("utf-8")
    yield b"type: done\ndata: [DONE]\n\n"

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring"""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    # Log response time
    process_time = time.time() - start_time
    logger.info(f"Response time: {process_time:.2f}s")
    
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Gemini connection
        test_response = model.generate_content("Hello", stream=False)
        return {
            "status": "healthy",
            "gemini_connection": "ok",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/send_message_stream")
async def send_message_stream(
    request: Request,
    text: str = Form(...),
    x_user_id: Optional[str] = Header(None, alias="x-user-id"),
    x_session_id: Optional[str] = Header(None, alias="x-session-id"),
    x_chat_id: Optional[str] = Header(None, alias="x-chat-id"),
    authorization: Optional[str] = Header(None),
):
    # Input validation
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty")
    
    if len(text) > 10000:  # Reasonable limit
        raise HTTPException(status_code=400, detail="Text too long (max 10000 characters)")
    
    # Log request details
    logger.info(f"Stream request from user {x_user_id}, session {x_session_id}")
    
    async def stream() -> AsyncGenerator[bytes, None]:
        stream_buffer = StreamBuffer()
        
        try:
            # Send initial progress
            yield b"type: progress\ndata: \xf0\x9f\x94\x80 Processing your request...\n\n"
            
            # Prepare messages for Gemini
            messages = [
                {"role": "user", "parts": [SYSTEM_INSTRUCTIONS]},
                {"role": "user", "parts": [text]}
            ]
            
            # Create Gemini stream with timeout and retry logic
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    gemini_stream = model.generate_content(
                        messages,
                        stream=True,
                        safety_settings=[
                            {
                                "category": "HARM_CATEGORY_HARASSMENT",
                                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                            },
                            {
                                "category": "HARM_CATEGORY_HATE_SPEECH",
                                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                            }
                        ]
                    )
                    break
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Gemini stream attempt {retry_count} failed: {e}")
                    
                    if retry_count >= max_retries:
                        raise StreamError(f"Failed to establish Gemini stream after {max_retries} attempts")
                    
                    # Brief delay before retry
                    await asyncio.sleep(1)
            
            # Process stream chunks with buffering
            chunk_count = 0
            
            for chunk in gemini_stream:
                try:
                    if hasattr(chunk, 'text') and chunk.text:
                        chunk_count += 1
                        
                        # Fix potential JSON issues in the chunk
                        fixed_chunk_text = extract_and_fix_json_chunks(chunk.text)
                        
                        # Add to buffer and get complete blocks
                        complete_blocks = stream_buffer.add_chunk(fixed_chunk_text)
                        
                        # Send complete blocks
                        for block in complete_blocks:
                            yield block.encode("utf-8")
                    
                    # Add small delay to prevent overwhelming the client
                    await asyncio.sleep(0.01)
                    
                except Exception as chunk_error:
                    logger.error(f"Error processing chunk: {chunk_error}")
                    yield f"type: error\ndata: Error processing response chunk\n\n".encode("utf-8")
                    break
            
            # Process any remaining buffer content
            remaining = stream_buffer.get_remaining()
            if remaining:
                yield remaining.encode("utf-8")
            
            # Log completion
            logger.info(f"Stream completed with {chunk_count} chunks")
            
            # Emit final done signal
            yield b"type: done\ndata: [DONE]\n\n"

        except StreamError as se:
            logger.error(f"Stream error: {se}")
            async for chunk in create_error_stream(str(se)):
                yield chunk
                
        except Exception as e:
            logger.error(f"Unexpected error in stream: {e}")
            async for chunk in create_error_stream("An unexpected error occurred"):
                yield chunk

    return StreamingResponse(
        stream(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return {"error": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )