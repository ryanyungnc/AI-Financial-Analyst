import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- 1. ALL MODELS (Blueprints) ---

# NPV Models
class YearCashFlow(BaseModel):
    year: int
    amount: float

class AnalysisRequest(BaseModel):
    text: str

# Concise Models
class Problem(BaseModel):
    question_type: int
    question_label: str
    incorrect_sentence: str

class GradeRequest(BaseModel):
    problem: Problem
    edited_sentence: str

# --- 2. THE APP SETUP ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change to your Vercel URL later!
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. THE ROUTES (The "Traffic Controller") ---

# HEALTH CHECK (Always keep this! Tells Render you are awake)
@app.get("/")
async def health_check():
    return {"status": "AI Server is Online", "projects": ["NPV Analyst", "Concise"]}

# --- CONCISE ENDPOINTS ---
@app.get("/concise/start")
async def start_game():
    # Call your function from your other file
    from concise_logic import initialize_concise 
    return initialize_concise()

@app.post("/concise/grade")
async def grade_sentence(request: GradeRequest):
    from concise_logic import give_feedback
    return give_feedback(request.problem, request.edited_sentence)

# --- NPV ENDPOINTS ---
@app.post("/npv/calculate")
async def handle_npv(request: AnalysisRequest):
    from npv_logic import text_analysis, calculate_npv, get_strategic_advice
    # ... your NPV logic here ...
    return {"status": "success"}