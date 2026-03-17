import os
import json
import sys
import math
import numpy_financial as npf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from scipy.optimize import brentq
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class YearCashFlow(BaseModel):
    year: int = Field(description = "The year number.")
    amount: float = Field(description = "The cash flow amount for the year")

class CashFlowData(BaseModel):
    discount_rate: float = Field(default = 0.1, description = "The numerical proportion of the cost of capital (e.g., 0.10 for 10%)")
    cash_flows: list[YearCashFlow] = Field(description = "List of cash for finite years.")
    perpetuity_amount: float = Field(default=0.0, description="The amount that continues forever AFTER the finite years end.")
    perpetuity_gr: float = Field(default = 0.0, description = "Growth rate if perpetuity w growth, else 0.0")

class StrategicAdvice(BaseModel):
    verdict: str = Field(description="A single 'Yes, you should undertake this investment' or 'No, you shouldn't undertake this investment' sentence with brief reasoning.")
    deep_dive: str = Field(description="A detailed paragraph explaining the financial justification.")
    suggestions: list[str] = Field(description="A list of 3 strings. Each must start with a bolded title sentence followed by reasoning.")

def text_analysis(file_path):
    #Initalize client
    client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))

    #Read message information
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            message_content = file.read()
    except FileNotFoundError:
        return None

    #Generate JSON content
    response = client.models.generate_content(
        model = "gemini-3.1-flash-lite-preview",
        config = types.GenerateContentConfig(
            system_instruction = """
                                    You are a financial data extractor. 
                                    1. Identify the 'Initial Investment' (Year 0).
                                    2. For recurring flows, calculate the NET flow per year. 
                                    (e.g., $1M profit - $100k support = $900k net flow).
                                    3. If 'in perpetuity' is mentioned, set perpetuity=True and 
                                    set perpetuity_gr=0 unless a growth rate is specified.
                                    4. Output ONLY the resulting JSON.
                                    """,
            response_mime_type = "application/json",
            response_schema = CashFlowData,
            temperature = 0.1
        ),
        contents = message_content
    )

    #Return parsed data
    return response.parsed

def calculate_npv(data):
    #Intialize
    total = 0.0
    r = data.discount_rate

    #Calculate NPV of finite cash flows
    for cf in data.cash_flows:
        total += cf.amount / ((1 + r) ** cf.year)

    #Calculate value of perpetuity, if any
    if(data.perpetuity):
        #Calculate value of entire perpetuity
        whole_perpetuity_value = calculate_perpetuity_cf(data, r)

        #Discount back to year 0
        total += whole_perpetuity_value / ((1 + r) ** data.cash_flows[-1].year)

    return total

def calculate_irr(data):
    sorted_cfs = sorted(data.cash_flows, key = lambda x: x.year)
    max_year = sorted_cfs[-1].year

    #Build list of CF
    cf_list = [0.0] * (max_year + 1)
    for cf in data.cash_flows:
        cf_list[cf.year] = cf.amount
    
    #Perpetuity case
    if (data.perpetuity):
        def npv_at_rate(r):
            if r <= data.perpetuity_gr:
                return float('inf')
            temp_data = data.model_copy(update={"discount_rate": r})
            return calculate_npv(temp_data)

        try:
            return brentq(npv_at_rate, 0.0001, 0.9999)
        except ValueError:
            return None
        
    #Check if perpetuity is valid for IRR calculation
    has_positive = any(x > 0 for x in cf_list)
    has_negative = any(x < 0 for x in cf_list)
    if not (has_positive and has_negative):
        return None

    #return IRR
    return npf.irr(cf_list)

def calculate_pi(data):
    r = data.discount_rate

    pv_inflows = 0.0
    pv_outflows = 0.0

    #Interate through data, adding to inflows and outflows
    for cf in data.cash_flows:
        pv = cf.amount / ((1 + r) ** cf.year)
        if cf.amount > 0:
            pv_inflows += pv
        else:
            pv_outflows += abs(pv)

    #Handle perpetuity inflows
    if data.perpetuity:
        term_val = calculate_perpetuity_cf(data, r)
        pv_term_val = term_val / ((1 + r) ** data.cash_flows[-1].year)
        if pv_term_val > 0:
            pv_inflows += pv_term_val
        else:
            pv_outflows += abs(pv_term_val)
    
    if pv_outflows == 0:
        return None

    return pv_inflows / pv_outflows

def calculate_payback_periods(data):
    #Check if applicable
    has_negative = any(cf.amount < 0 for cf in data.cash_flows)
    if not has_negative:
        return None, None

    sorted_cfs = sorted(data.cash_flows, key = lambda x: x.year)

    simple_balance = 0.0
    discounted_balance = 0.0
    r = data.discount_rate

    payback_simple = None
    payback_discounted = None

    #Iterate through finite years
    for cf in sorted_cfs:
        simple_balance += cf.amount
        discounted_balance += cf.amount / ((1 + r) ** cf.year)

        if payback_simple is None and simple_balance >= 0:
            payback_simple = cf.year
        if payback_discounted is None and discounted_balance >= 0:
            payback_discounted = cf.year
    
    #If not paid back yet and perpetuity
    if data.perpetuity:
        last_year = data.cash_flows[-1].year
        perp_cf = sorted_cfs[-1].amount
        g = data.perpetuity_gr
        c = perp_cf * (1 + g)

        if payback_simple is None and perp_cf > 0:
            target = abs(simple_balance)

            if g == 0:
                extra_years = target / perp_cf
            else:
                val_to_log = (target * g / perp_cf) + 1
                extra_years = math.log(val_to_log) / math.log(1 + g)
            
            payback_simple = last_year + extra_years

        if payback_discounted is None and r > g:
            target = abs(discounted_balance)
            target_at_last_year = target * ((1 + r) ** last_year)

            #Logic check that payments grow faster than debt
            factor = (target_at_last_year * (r - g)) / c

            if factor < 1:
                numerator = math.log(1 - factor)
                denominator = math.log((1 + g) / (1 + r))
                payback_discounted = last_year + (numerator / denominator)

    return payback_simple, payback_discounted


def calculate_perpetuity_cf(data, r):
    #Safety check: g must be less than r
    if (data.perpetuity_gr >= r):
        print("Warning: Growth rate >= discount rate. Perpetuity value is undefined.")
        return 0.0

    last_cf = data.cash_flows[-1]
    g = data.perpetuity_gr

    #Calculate value of first CF after finite series
    next_cf_amount = last_cf.amount * (1 + g)

    #Return CF value of entire perpetuity
    return (next_cf_amount / (r - g))

def calculate_terminal_value(data, r):
    if data.perpetuity_gr >= r:
        return 0.0
    
    last_cf = data.cash_flows[-1].amount
    next_cf_amount = last_cf * (1 + data.perpetuity_gr)

    return next_cf_amount / (r - data.perpetuity_gr)

def get_strategic_advice(data, results):
    client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))

    irr_str = f"{results['irr']:.2%}" if results['irr'] is not None else "N/A"
    pb_d_str = f"{results['payback_d']:.2f}" if results['payback_d'] is not None else "N/A"

    context = (
        f"Analyze this investment for our firm: NPV=${results['npv']:,.2f}, "
        f"IRR={irr_str}, "
        f"Discounted Payback={pb_d_str} years, "
        f"Discount rate={data.discount_rate}. "
        "Provide a structured executive summary."
    )

    response = client.models.generate_content(
        model = "gemini-3.1-flash-lite-preview",
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a Senior Financial Controller. Provide advice in three parts: "
                "1. A clear Verdict (Yes/No + reason). "
                "2. A Deep Dive analysis into the risk and return. "
                "3. Three actionable suggestions. Format each suggestion as: "
                "Title Sentence. Reasoning text."
            ),
            response_mime_type="application/json",
            response_schema=StrategicAdvice,
        ),
        contents=context
    )

    return response.parsed




if __name__ == "__main__":
    target = "testing/ex1.txt"
    print(f"Extracting CF data from {target}...")

    #Data parsing
    data = text_analysis(target)
    if data:
        print(f"====\nDiscount Rate: {data.discount_rate}")
        print(f"Finite Cash Flows:")
        for cf in data.cash_flows:
            print(f"    Year {cf.year}: ${cf.amount:,.2f}")
        
        if(data.perpetuity == True):
            print(f"This is a perpetuity with growth rate: {data.perpetuity_gr}")
        else:
            print("Not a perpetuity")
    else:
        print("Failed to extract data.")
        sys.exit(1)

    #NPV, IRR, PI data
    npv = calculate_npv(data)
    irr = calculate_irr(data)
    pi = calculate_pi(data)
    pb_s, pb_d = calculate_payback_periods(data)
    results = {
        "npv" : npv,
        "irr" : irr,
        "pi" : pi,
        "payback_s" : pb_s,
        "payback_d" : pb_d
    }

    print(f"====\nNet Present Value (NPV): ${npv:,.2f}")

    if irr is not None:
        print(f"Internal Rate of Return (IRR): {irr:.2%}")
    else:
        print("IRR could not be calculated")
    if pi is not None:
        print(f"Profibility Index (PI): {pi:.2f}")
    else:
        print("Profibility Index could not be calculated")

    if pb_s is not None:
        print(f"Payback Period: {pb_s:.2f} years")
    else:
        print("Payback period could not be calculated")
    if pb_d is not None:
        print(f"Discounted Payback Period: {pb_d:.2f} years")
    else:
        print("Discounted payback period could not be calculated")

    print(f"====\nAI Strategic Summary")
    advice = get_strategic_advice(data, results)
    print(advice)



#FastAPI Integration

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change to site address later
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    text: str

@app.post("/calculate")
async def handle_analysis(request: AnalysisRequest):
    try:
        #Conversion to text file for function
        with open("temp_input.txt", "w") as f:
            f.write(request.text)
        
        data = text_analysis("temp_input.txt")

        if not data:
            raise HTTPException(status_code=400, detail="Could not extract data.")

        #Function calculations
        npv = calculate_npv(data)
        irr = calculate_irr(data)
        pi = calculate_pi(data)
        pb_s, pb_d = calculate_payback_periods(data)
        
        results = {
            "npv": npv,
            "irr": irr,
            "pi": pi,
            "payback_s": pb_s,
            "payback_d": pb_d
        }

        # Get the strategic advice
        advice_obj = get_strategic_advice(data, results)

        # Return everything as JSON to Next.js
        return {
            "npv": npv,
            "irr": irr,
            "pi": pi,
            "payback_period_s": pb_s,
            "payback_period_d": pb_d,
            "advice": advice_obj.model_dump(),
            "raw_cash_flows": [cf.model_dump() for cf in data.cash_flows],
            "is_perpetuity": data.perpetuity
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run it with: uvicorn npv_machine:app --reload
