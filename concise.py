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


class Problem(BaseModel):
    question_type: int = Field(description = "The integer ID of this question type (1-6)")
    question_label: str = Field(description = "The full instruction string for this question type")
    incorrect_sentence: str = Field(description = "The flawed sentence the student must correct")

class ProblemSet(BaseModel):
    problems: list[Problem] 

class Feedback(BaseModel):
    objective_score: int = Field(description = "Points scored based on how well they fixed the problem without introducing new errors (0-5)") 
    sentence_score: int = Field(description = "Points scored based on meaning preservation and conciseness (0-5)")
    total_score: int = 0
    reasoning: str = Field(description = "Two sentences max. First sentence explains the objective score, second explains the sentence score. Be specific about what the student did or missed.")
    ideal_sentence: str = Field(description="One strong rewrite of the original flawed sentence that fully addresses the question type. Frame as a possible solution, not the only solution.")

def initialize_concise():
    response = client.models.generate_content(
        model = "gemini-3.1-flash-lite-preview",
        config = types.GenerateContentConfig(
            system_instruction = """
                You are a business writing coach generating practice problems for professionals.

                SENTENCE REQUIREMENTS:
                - Each sentence must be 15-25 words long
                - Set in a realistic business context (emails, reports, meeting notes, proposals)
                - Contain exactly one flaw matching its question type — no other writing errors
                - Sound like something a real (but flawed) writer would actually write

                QUESTION TYPE DEFINITIONS:
                1: Active voice — subject is hidden, action is carried by a "to be" verb + past participle
                2: "To be" verbs — overuse of is/are/am/be/been/being/seems/appears where a stronger verb exists
                3: False subjects — sentence opens with "It is", "There is/are", "This is", "Those are"
                4: Camouflaged verbs — a strong verb is buried as a noun ending in -ment, -ency, or -tion
                5: Plain language — contains jargon, legalese, or unnecessarily complex vocabulary
                6: Wordiness — contains redundant phrases like "due to the fact that" or "at this point in time"

                OUTPUT RULES:
                - Select exactly 10 question types, chosen randomly across all 6 types
                - No question type should appear more than 3 times
                - question_label must be the full instruction string for that type (e.g. "Change the sentence to active voice.")
                """,
            response_mime_type = "application/json",
            response_schema = ProblemSet
        ),
        contents = "Generate the problem set."
    )

    return response.parsed

def give_feedback(problem, edited_sentence):
    response = client.models.generate_content(
        model = "gemini-3.1-flash-lite-preview",
        config = types.GenerateContentConfig(
            system_instruction = """
                You are a business writing coach grading practice problems for professionals. Analyize the question that was asked, response, and assign an objective score, sentence score, and reasoning for the scores.
                
                QUESTION TYPE DEFINITIONS:
                1: Active voice — subject is hidden, action is carried by a "to be" verb + past participle
                2: "To be" verbs — overuse of is/are/am/be/been/being/seems/appears where a stronger verb exists
                3: False subjects — sentence opens with "It is", "There is/are", "This is", "Those are"
                4: Camouflaged verbs — a strong verb is buried as a noun ending in -ment, -ency, or -tion
                5: Plain language — contains jargon, legalese, or unnecessarily complex vocabulary
                6: Wordiness — contains redundant phrases like "due to the fact that" or "at this point in time"

                OBJECTIVE SCORE CRITERIA:
                - Did they fix the given sentence flaw?
                - Did they do so without introducing new errors?

                SENTENCE SCORE CRITERIA:
                - Did they preserve the meaning of the original sentence?
                - Is it concise?
                - Does it sound natural?

                OUTPUT RULES:
                - Objective and sentence scores should be graded on the scale of 0 = many mistakes, 1-2 = some mistakes, 3 = relatively well, 4 = no mistakes but not the best solution, 5 = perfect
                - Sentence reasoning should use objective wording and address the writer using second-person pronouns.
                - ideal_sentence should fix only the target flaw, not rewrite the entire sentence unnecessarily
                """,
            response_mime_type = "application/json",
            response_schema = Feedback
        ),
        contents = f"""
            ORIGINAL PROBLEM:
            Question Type: {problem.question_type}
            Instruction: {problem.question_label}
            Flawed Sentence: {problem.incorrect_sentence}

            STUDENT SUBMISSION:
            {edited_sentence}
            """
    )

    feedback = response.parsed
    feedback.total_score = feedback.objective_score + feedback.sentence_score

    return feedback


if __name__ == "__main__":
    problem_set = initialize_concise()
    if problem_set:
        for problem in problem_set.problems:
            print(problem.question_label)
            print(problem.incorrect_sentence)
            champ = input(f"Corrected Sentence: ")

            feedback = give_feedback(problem, champ)
            print(feedback.objective_score, feedback.sentence_score, feedback.total_score)
            print(feedback.reasoning)
            print(feedback.ideal_sentence)
            print("===\n\n")
    else:
        print("Failed to extract problem set")
        sys.exit(1)
    
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/concise/start")
async def start_game():
    try:
        problem_set = initialize_concise()
        return problem_set
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class GradeRequest(BaseModel):
    problem: Problem
    edited_sentence: str

@app.post("/concise/grade")
async def grade_sentence(request: GradeRequest):
    try:
        feedback = give_feedback(request.problem, request.edited_sentence)
        return feedback
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))