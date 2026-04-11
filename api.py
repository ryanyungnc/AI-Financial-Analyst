import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    text: str

@app.post("/calculate")
async def handle_analysis(request: AnalysisRequest):
    try:
        data = text_analysis_from_string(request.text) 
        
        if not data:
            raise HTTPException(status_code=400, detail="Gemini couldn't find data.")

        npv = calculate_npv(data)
        irr = calculate_irr(data)
        pi = calculate_pi(data)
        pb_s, pb_d = calculate_payback_periods(data)
        
        results = {"npv": npv, "irr": irr, "pi": pi, "payback_d": pb_d}
        
        advice = get_strategic_advice(data, results)

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