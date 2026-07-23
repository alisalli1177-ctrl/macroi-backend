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
        content.append({"type": "image_url", "image_url": {"url": req.base64_image, "detail": "low"}})
        
    if req.base64_images:
        for img in req.base64_images:
            if len(img) > 10:
                content.append({"type": "image_url", "image_url": {"url": img, "detail": "low"}})

    payload = {
        "model": req.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 500,
        "temperature": 0.2
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
            # First try direct parse
            parsed_direct = json.loads(ai_text)
            if isinstance(parsed_direct, list):
                return parsed_direct
        except json.JSONDecodeError:
            pass

        try:
            json_match = re.search(r'\{[^{}]*\}', ai_text)
            if json_match:
                parsed = json.loads(json_match.group())
                if "calories" in parsed and "protein" in parsed:
                    return parsed
        except (json.JSONDecodeError, KeyError):
            pass

        try:
            json_array_match = re.search(r'\[.*\]', ai_text, re.DOTALL)
            if json_array_match:
                parsed = json.loads(json_array_match.group())
                if isinstance(parsed, list):
                    return parsed
        except (json.JSONDecodeError, KeyError):
            pass
        
        # For non-JSON responses (like form analysis), return full response
        return data
        
    except requests.exceptions.HTTPError as he:
        raise HTTPException(status_code=he.response.status_code, detail=he.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------
# AI Kitchen – Recipe Generation Endpoint
# Takes ingredients + preferences, returns 3 healthy recipes.
# -------------------------------------------------------------

class RecipeRequest(BaseModel):
    ingredients: List[str]
    meal_type: Optional[str] = "Lunch"
    cooking_time: Optional[str] = "< 30 min"
    goal: Optional[str] = "Balanced"

@app.post("/generate-recipes")
def generate_recipes(req: RecipeRequest):
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        raise HTTPException(status_code=500, detail="OpenAI API key missing from backend .env")

    ingredients_str = ", ".join(req.ingredients)

    prompt = f"""You are a professional chef and nutritionist. The user has these ingredients: {ingredients_str}.
Meal type: {req.meal_type}. Max cooking time: {req.cooking_time}. Nutrition goal: {req.goal}.

Generate exactly 3 healthy recipes using ONLY these ingredients (plus common pantry staples like salt, pepper, oil).
For each recipe provide accurate calorie and macro estimates.

Reply ONLY with a valid JSON array, no markdown:
[{{"name":"Recipe Name","emoji":"🍝","calories":510,"protein":42,"carbs":45,"fat":15,"cookingTime":"20 min","rating":4.8,"ingredients":["200g Chicken","100g Pasta"],"steps":["Boil pasta in salted water.","Cook chicken in olive oil.","Mix together and serve."]}}]"""

    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": 1500,
        "temperature": 0.7
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=45
        )
        response.raise_for_status()
        data = response.json()

        ai_text = data["choices"][0]["message"]["content"].strip()

        # Parse the JSON array from the response
        try:
            # Try direct parse first
            recipes = json.loads(ai_text)
            if isinstance(recipes, list):
                return recipes
        except json.JSONDecodeError:
            pass

        # Try to extract JSON array from text
        try:
            json_match = re.search(r'\[.*\]', ai_text, re.DOTALL)
            if json_match:
                recipes = json.loads(json_match.group())
                if isinstance(recipes, list):
                    return recipes
        except (json.JSONDecodeError, KeyError):
            pass

        raise HTTPException(status_code=500, detail="Failed to parse recipe response from AI")

    except requests.exceptions.HTTPError as he:
        raise HTTPException(status_code=he.response.status_code, detail=he.response.text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))