import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# ... (Keep all your existing imports: math, genai, etc.)

app = FastAPI()

# This part allows your Next.js "Waiter" to talk to the Python "Chef"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In development, this allows any site to talk to your API
    allow_methods=["*"],
    allow_headers=["*"],
)

# This is what the Next.js site will send to Python
class AnalysisRequest(BaseModel):
    text: str

@app.post("/calculate")
async def handle_analysis(request: AnalysisRequest):
    try:
        # 1. Run your Gemini extraction using the text sent from the website
        # (You'll wrap your existing logic here)
        data = text_analysis_from_string(request.text) 
        
        if not data:
            raise HTTPException(status_code=400, detail="Gemini couldn't find data.")

        # 2. Run all the math functions we wrote
        npv = calculate_npv(data)
        irr = calculate_irr(data)
        pi = calculate_pi(data)
        pb_s, pb_d = calculate_payback_periods(data)
        
        results = {"npv": npv, "irr": irr, "pi": pi, "payback_d": pb_d}
        
        # 3. Get the AI advice
        advice = get_strategic_advice(data, results)

        # 4. Send it all back to Next.js
        return {
            "npv": npv,
            "irr": irr,
            "pi": pi,
            "payback_period": pb_d,
            "advice": advice
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run this with: uvicorn api:app --reload