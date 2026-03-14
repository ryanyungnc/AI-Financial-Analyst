import os
from google import genai
from dotenv import load_dotenv

# 1. Load the .env file
load_dotenv()

def summarize_file(file_path):
    # 2. Initialize the Client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # 3. Read the local file
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
    except FileNotFoundError:
        return f"Error: {file_path} not found!"

    # 4. Use a model from YOUR specific list
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=f"Please provide a concise, bulleted summary of this text:\n\n{text_content}"
    )
    
    return response.text

if __name__ == "__main__":
    target = "testing/notes.txt"
    print(f"--- Summarizing {target} ---")
    print(summarize_file(target))