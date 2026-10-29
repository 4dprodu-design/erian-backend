from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import edge_tts
import io
import os
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str
    voice: str = "en-US-AriaNeural"
    rate: str = "-25%"

class ChatRequest(BaseModel):
    text: str
    system_prompt: str = "You are Sofia, a friendly English tutor."

@app.post("/api/tts")
async def generate_tts(request: TextRequest):
    try:
        communicate = edge_tts.Communicate(request.text, request.voice, rate=request.rate)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return StreamingResponse(io.BytesIO(audio_data), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_with_deepseek(request: ChatRequest):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return {"response": "Error: API Key no configurada en Vercel."}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.text}
        ],
        "temperature": 0.7
    }

    async with httpx.AsyncClient() as client:
        response = await client.post("https://api.deepseek.com/chat/completions", json=payload, headers=headers)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error en DeepSeek")
    
    data = response.json()
    return {"response": data["choices"][0]["message"]["content"]}
