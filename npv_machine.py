import os
import json
import sys
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
    perpetuity: bool = Field(description = "True if perpetuity, else false")
    perpetuity_gr: float = Field(default = 0.0, description = "Growth rate if perpetuity w growth, else 0.0")


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
            system_instruction = "You are a financial analyst assistant. Extract cash flow data from the user's message, calculating for finite years and extracting perpetuity data for infinite years.",
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
        #Safety check: g must be less than r
        if (data.perpetuity_gr >= r):
            print("Warning: Growth rate >= discount rate. Perpetuity value is undefined.")
            return total

        last_cf = data.cash_flows[-1]
        n = last_cf.year
        g = data.perpetuity_gr

        #Calculate value of first CF after finite series
        next_cf_amount = last_cf.amount * (1 + g)

        #Calculate value of entire perpetuity
        whole_perpetuity_value = next_cf_amount / (r - g)

        #Discount back to year 0
        total += whole_perpetuity_value / ((1 + r) ** n)

    return total




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

    #NPV data
    npv = calculate_npv(data)
    print(f"====\n Project NPV: ${npv:,.2f}")

    
