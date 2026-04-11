import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

#Show & Tell class model
class MonroeRequest(BaseModel):
    attention: str
    challenge: str
    solution: str
    benefits: str
    call_to_action: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "AI Server is Online", "projects": ["NPV Analyst", "Concise", "Exec File"]}


@app.get("/concise/start")
async def start_game():
    try:
        from concise import initialize_concise 
        problem_set = initialize_concise()
        return problem_set
    except Exception as e:
        print(f"Error in start_game: {e}") 
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/concise/grade")
async def grade_sentence(request: GradeRequest):
    try:
        from concise import give_feedback
        feedback = give_feedback(request.problem, request.edited_sentence)
        return feedback
    except Exception as e:
        print(f"Error in grade_sentence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/npv/calculate")
async def handle_npv(request: AnalysisRequest):
    try:
        from npv_machine import (
            text_analysis,
            calculate_npv,
            calculate_irr,
            calculate_pi,
            calculate_payback_periods,
            get_strategic_advice,
            validate_cash_flow_data
        )

        with open("temp_input.txt", "w") as f:
            f.write(request.text)
        
        data = text_analysis("temp_input.txt")
        if not data:
            raise HTTPException(status_code=400, detail="Could not extract data.")

        validation_error = validate_cash_flow_data(data)
        if validation_error:
            raise HTTPException(status_code=422, detail=validation_error)

        results = {
            "npv": calculate_npv(data),
            "irr": calculate_irr(data),
            "pi": calculate_pi(data),
            "payback_s": calculate_payback_periods(data)[0],
            "payback_d": calculate_payback_periods(data)[1]
        }

        advice_obj = get_strategic_advice(data, results)

        return {
            "npv": results["npv"],
            "irr": results["irr"],
            "pi": results["pi"],
            "payback_period_s": results["payback_s"],
            "payback_period_d": results["payback_d"],
            "advice": advice_obj.model_dump(),
            "raw_cash_flows": [cf.model_dump() for cf in data.cash_flows],
            "is_perpetuity": data.is_perpetuity
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"NPV Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/monroe/grade")
async def generate_feedback(request: MonroeRequest):
    try:
        from busi401assignment import give_feedback
        feedback = give_feedback(request)
        return feedback

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Monroe Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))