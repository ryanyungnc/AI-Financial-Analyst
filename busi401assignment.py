import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))

class AnalysisRequest(BaseModel):
    attention: str
    challenge: str
    solution: str
    benefits: str
    call_to_action: str

class Feedback(BaseModel):
    attention_score: int
    a_feedback: str
    challenge_score: int
    c_feedback: str
    solution_score: int
    s_feedback: str
    benefits_score: int
    b_feedback: str
    call_score: int
    call_feedback: str
    total_score: int = 0
    

def give_feedback(request):
    response = client.models.generate_content(
        model = "gemini-3.1-flash-lite-preview",
        config = types.GenerateContentConfig(
            system_instruction = """
                You are a Business Writing Coach. Grade professional writing using Monroe's Motivated Sequence.

                GRADING RUBRIC (Total 20 points per section):
                1. Portion Criteria (10 pts): Did they include the core components of the specific step? 
                2. Style Criteria (10 pts): Is it concise (1-2 sentences), natural-sounding, and contextually appropriate?

                SCALING:
                - 0-3: Low effort/Missing
                - 4-5: Below average/Weak logic
                - 6-8: Average/Clear but basic
                - 9-10: Great/Compelling and professional

                STEPS TO EVALUATE:
                1. Attention: Grab the reader's interest.
                2. Need (Challenge): Define the problem clearly.
                3. Satisfaction (Solution): Provide a viable fix.
                4. Visualization (Benefits): Help the reader see the results.
                5. Action: Specific call to action.

                OUTPUT RULES:
                - Provide a score for each of the 5 portions.
                - Each portion's score must be the sum of Portion Criteria + Style Criteria.
                - Feedback must be 1-2 sentences per category, focusing on actionable improvement.
                """,
            response_mime_type = "application/json",
            response_schema = Feedback
        ),
        contents = f"""
            STUDENT's INPUTTED MONROES MOTIVATED SEQUENCE:
            Attention: {request.attention}
            Challenge: {request.challenge}
            Solution: {request.solution}
            Benefits: {request.benefits}
            Call to Action: {request.call_to_action}
            """
    )

    feedback = response.parsed
    feedback.total_score = feedback.attention_score + feedback.challenge_score + feedback.solution_score + feedback.benefits_score + feedback.call_score

    return feedback


if __name__ == "__main__":
    request = AnalysisRequest(
        attention = input("Attention:"),
        challenge = input("Challenge:"),
        solution = input("Solution:"),
        benefits = input("Benefits:"),
        call_to_action = input("Call to Action:")
    )

    feedback = give_feedback(request)

    print(feedback.attention_score)
    print(feedback.a_feedback)
    print(feedback.challenge_score)
    print(feedback.c_feedback)
    print(feedback.solution_score)
    print(feedback.s_feedback)
    print(feedback.benefits_score)
    print(feedback.b_feedback)
    print(feedback.call_score)
    print(feedback.call_feedback)
    print(feedback.total_score)

