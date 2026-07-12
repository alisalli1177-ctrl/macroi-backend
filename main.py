import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import json
import re

load_dotenv()

app = FastAPI()

@app.get("/")
def home():
    return {"mesaj": "MacroAI Backend is active!"}

# -------------------------------------------------------------
# Secure Vision & Text Analysis Endpoint (OpenAI GPT-4o)
# All API keys stay on the server - the app never sees them.
# -------------------------------------------------------------
from typing import Optional, List

class VisionRequest(BaseModel):
    base64_image: Optional[str] = None
    base64_images: Optional[List[str]] = None
    prompt: str
    model: Optional[str] = "gpt-4o-mini"

@app.post("/analyze-vision")
def analyze_vision(req: VisionRequest):
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        raise HTTPException(status_code=500, detail="OpenAI API key missing from backend .env")

    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }

    # Build messages based on whether we have an image or not
    content = [{"type": "text", "text": req.prompt}]
    
    if req.base64_image and len(req.base64_image) > 10:
        content.append({"type": "image_url", "image_url": {"url": req.base64_image}})
        
    if req.base64_images:
        for img in req.base64_images:
            if len(img) > 10:
                content.append({"type": "image_url", "image_url": {"url": img}})

    payload = {
        "model": req.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 800,
        "temperature": 0.3
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        ai_text = data["choices"][0]["message"]["content"].strip()
        
        # Try to parse as JSON (for food analysis responses)
        try:
            json_match = re.search(r'\{[^{}]*\}', ai_text)
            if json_match:
                parsed = json.loads(json_match.group())
                if "calories" in parsed and "protein" in parsed:
                    return parsed
        except (json.JSONDecodeError, KeyError):
            pass
        
        # For non-JSON responses (like form analysis), return full response
        return data
        
    except requests.exceptions.HTTPError as he:
        raise HTTPException(status_code=he.response.status_code, detail=he.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))