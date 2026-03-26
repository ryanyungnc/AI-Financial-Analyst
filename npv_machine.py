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
client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))


class YearCashFlow(BaseModel):
    year: int = Field(description = "The year number.")
    amount: float = Field(description = "The cash flow amount for the year")

class CashFlowData(BaseModel):
    discount_rate: float = Field(default = 0.1, description = "The numerical proportion of the cost of capital (e.g., 0.10 for 10%)")
    cash_flows: list[YearCashFlow] = Field(description = "List of cash for finite years.")
    is_perpetuity: bool = Field(default=False)
    perpetuity_amount: float = Field(default=0.0, description="The amount that continues forever AFTER the finite years end.")
    perpetuity_gr: float = Field(default = 0.0, description = "Growth rate if perpetuity w growth, else 0.0")

class StrategicAdvice(BaseModel):
    verdict: str = Field(description="A single 'Yes, you should undertake this investment' or 'No, you shouldn't undertake this investment' sentence with brief reasoning.")
    deep_dive: str = Field(description="A detailed paragraph explaining the financial justification.")
    suggestions: list[str] = Field(description="A list of 3 strings. Each starts with a 5-15 word bolded title sentence with the recomendation followed by 2 unbolded sentences elaborating on the reasoning.")

def text_analysis(file_path):
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

                                    CRITICAL TIMING RULES:
                                    - Costs paid today = Year 0
                                    - If construction takes 1 year, Year 1 cash flow = $0 (plant not yet operational)
                                    - First operating cash flow begins the year AFTER construction completes
                                    - '20 years after completion' with completion at Year 1 means operations Year 2-21, shutdown Year 22

                                    CRITICAL AMOUNT RULES:
                                    - NEVER net two cash flows that occur in the same year
                                    - If a final operating payment AND a shutdown cost both occur at Year 22,
                                    create TWO separate entries and sum them: (15000000) + (-200000000) = -185000000
                                    OR represent as one entry: {year: 22, amount: -185000000}
                                    - Shutdown/cleanup costs are ALWAYS their own cash flow, never merged silently
                                    -If a shutdown cost and operating payment occur in the same year, 
                                    the year's cash flow = operating + shutdown (e.g. 15M + (-200M) = -185M). 
                                    The shutdown cost itself must always be a round number as stated in the problem.

                                    Output ONLY the resulting JSON.
                                    """,
            response_mime_type = "application/json",
            response_schema = CashFlowData,
            temperature = 0.1
        ),
        contents = message_content
    )

    #Return parsed data
    return response.parsed

def validate_cash_flow_data(data: CashFlowData) -> str | None:
    """Returns an error message string if data is invalid, else None."""
    if not data.cash_flows:
        return "No cash flows could be extracted from the input."
    
    # Must have at least one Year 0 negative (investment) or some structure
    if len(data.cash_flows) < 2:
        return "Input must describe at least an initial investment and one return period."
    
    # Discount rate sanity check
    if not (0 < data.discount_rate < 1):
        return f"Extracted discount rate ({data.discount_rate}) is not a valid proportion (must be between 0 and 1)."
    
    # All amounts zero is a red flag
    if all(cf.amount == 0 for cf in data.cash_flows):
        return "All extracted cash flows are zero — please provide a valid investment scenario."
    
    return None  # All good

def calculate_npv(data):
    #Intialize
    total = 0.0
    r = data.discount_rate

    #Calculate NPV of finite cash flows
    for cf in data.cash_flows:
        total += cf.amount / ((1 + r) ** cf.year)

    #Calculate value of perpetuity, if any
    if(data.is_perpetuity):
        last_year = data.cash_flows[-1].year
        terminal_value = calculate_perpetuity_cf(data, r)

        #Discount back to year 0 and add
        total += terminal_value / ((1 + r) ** last_year)

    return total

def calculate_irr(data):
    sorted_cfs = sorted(data.cash_flows, key = lambda x: x.year)
    max_year = sorted_cfs[-1].year

    #Build list of CF
    cf_list = [0.0] * (max_year + 1)
    for cf in data.cash_flows:
        cf_list[cf.year] = cf.amount
    
    #Perpetuity case
    if (data.is_perpetuity):
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
    if data.is_perpetuity:
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
    if data.is_perpetuity:
        last_year = sorted_cfs[-1].year  

        perp_cf = data.perpetuity_amount if data.perpetuity_amount > 0 else sorted_cfs[-1].amount
        g = data.perpetuity_gr
        c = perp_cf * (1 + g)  # first perpetuity payment

        if payback_simple is None and perp_cf > 0:
            target = abs(simple_balance)

            if g == 0:
                extra_years = target / perp_cf
            else:
                val_to_log = (target * g / perp_cf) + 1
                
                if val_to_log <= 0:
                    payback_simple = None
                else:
                    extra_years = math.log(val_to_log) / math.log(1 + g)
                    payback_simple = last_year + extra_years

            if g == 0:
                payback_simple = last_year + extra_years

        if payback_discounted is None and r > g:
            target = abs(discounted_balance)
            target_at_last_year = target * ((1 + r) ** last_year)

            factor = (target_at_last_year * (r - g)) / c

            if 0 < factor < 1:
                numerator = math.log(1 - factor)
                denominator = math.log((1 + g) / (1 + r))
                payback_discounted = last_year + (numerator / denominator)

    return payback_simple, payback_discounted


def calculate_perpetuity_cf(data, r):
    g = data.perpetuity_gr
    if g >= r:
        return 0.0
    
    # Use the explicit amount if provided, otherwise fallback to last_cf
    base_amount = data.perpetuity_amount if data.perpetuity_amount > 0 else data.cash_flows[-1].amount
    
    # Perpetuity formula: Next Year's Flow / (r - g)
    next_year_cf = base_amount * (1 + g)
    return next_year_cf / (r - g)

def calculate_terminal_value(data, r):
    if data.perpetuity_gr >= r:
        return 0.0
    
    last_cf = data.cash_flows[-1].amount
    next_cf_amount = last_cf * (1 + data.perpetuity_gr)

    return next_cf_amount / (r - data.perpetuity_gr)

def get_strategic_advice(data, results):
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
                "Title Sentence. Reasoning text. They will be plain text, no puctuation like astericks"
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
        
        if(data.is_perpetuity == True):
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
            "is_perpetuity": data.is_perpetuity
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run it with: uvicorn npv_machine:app --reload
