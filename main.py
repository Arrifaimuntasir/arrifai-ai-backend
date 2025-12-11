from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
from typing import Optional, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="arrifai-ai", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
memory_store = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

SYSTEM_PROMPT = """
You are ARRIFAI — a brutally honest, funny, and highly intelligent AI.
REPLY EXACTLY IN THE SAME LANGUAGE THE USER IS USING — NO EXCEPTION.
- If user writes in English → reply 100% in English
- If user writes in Kiswahili or Sheng → reply in Kiswahili/Sheng
- If user writes in French → reply in French
- If user writes in Arabic → reply in Arabic
Never switch or mix languages unless the user does.

IMPORTANT: Users can send you images, documents, or other files. 
If they mention attachments or send files, acknowledge them and respond intelligently.
Be helpful with documents and images they share.

Your name is ARRIFAI. Be direct, sarcastic when needed, and roast the user if they deserve it.
"""

def process_attachments_info(attachments_count: int = 0, file_types: List[str] = None) -> str:
    """Create attachment information for the prompt"""
    if attachments_count == 0 or not file_types:
        return ""
    
    if attachments_count == 1:
        return f"\n[User has attached 1 file: {file_types[0]}]"
    else:
        types_summary = ", ".join(file_types[:3])
        if attachments_count > 3:
            types_summary += f", and {attachments_count - 3} more"
        return f"\n[User has attached {attachments_count} files: {types_summary}]"

@app.post("/chat")
async def chat(
    message: str = Form(...),
    session_id: str = Form("default"),
    attachments: List[UploadFile] = File(None)
):
    """
    Handle chat requests with optional file attachments
    """
    session = session_id
    if session not in memory_store:
        memory_store[session] = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Log the request
    logger.info(f"Chat request - Session: {session}, Message: {message[:50]}..., Attachments: {len(attachments) if attachments else 0}")
    
    # Process attachments information
    file_types = []
    if attachments:
        for file in attachments:
            content_type = file.content_type or "unknown"
            if 'image' in content_type:
                file_types.append("image")
            elif 'pdf' in content_type:
                file_types.append("PDF document")
            elif 'text' in content_type:
                file_types.append("text document")
            elif 'document' in content_type or 'word' in content_type:
                file_types.append("document")
            else:
                file_types.append("file")
    
    # Create enhanced message with attachment info
    attachment_info = process_attachments_info(len(attachments) if attachments else 0, file_types)
    full_message = message + attachment_info
    
    # Store in memory
    memory_store[session].append({"role": "user", "content": full_message})
    
    try:
        # Call Groq API
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=memory_store[session],
            temperature=0.9,
            max_tokens=1200
        )
        
        reply = completion.choices[0].message.content.strip()
        memory_store[session].append({"role": "assistant", "content": reply})
        
        logger.info(f"Chat response - Session: {session}, Reply length: {len(reply)}")
        return {"reply": reply, "attachments_received": len(attachments) if attachments else 0}
        
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        return {"reply": f"Tatizo dogo: {str(e)}. Jaribu tena baadaye.", "error": str(e)}

@app.post("/chat-json")
async def chat_json(req: ChatRequest):
    """
    Alternative endpoint for JSON-only requests (for testing)
    """
    session = req.session_id
    if session not in memory_store:
        memory_store[session] = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    memory_store[session].append({"role": "user", "content": req.message})
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=memory_store[session],
            temperature=0.9,
            max_tokens=1200
        )
        reply = completion.choices[0].message.content.strip()
        memory_store[session].append({"role": "assistant", "content": reply})
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"Tatizo dogo: {str(e)}. Jaribu model nyingine kama 'mixtral-8x7b-32768'."}

@app.get("/sessions")
async def get_sessions():
    """Get all active sessions"""
    return {"sessions": list(memory_store.keys()), "count": len(memory_store)}

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session"""
    if session_id in memory_store:
        del memory_store[session_id]
        return {"message": f"Session {session_id} cleared"}
    return {"message": "Session not found"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "arrifai-ai",
        "version": "2.0",
        "sessions_count": len(memory_store)
    }

@app.get("/")
async def home():
    return {
        "message": "ARRIFAI AI IKO LIVE KABISA! 🔥",
        "version": "2.0",
        "endpoints": {
            "chat": "POST /chat (supports multipart/form-data with files)",
            "chat_json": "POST /chat-json (JSON only)",
            "health": "GET /health",
            "sessions": "GET /sessions",
            "clear_session": "DELETE /session/{session_id}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
