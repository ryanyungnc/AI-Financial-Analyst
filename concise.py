import os
from google import genai
from google.genai import types

load_dotenv()

def concise_coach(file_path):
    client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))
