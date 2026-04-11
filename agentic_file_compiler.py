import os
import json
import time
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel
from enum import Enum

load_dotenv()
client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))

class BlockType(str, Enum):
    heading = "heading"
    subheading = "subheading"
    text = "text"
    image_prompt = "image_prompt"

class Block(BaseModel):
    type: BlockType
    content: str

class Document(BaseModel):
    title: str
    blocks: list[Block]

def prepare_file(path):
    if path.endswith(".xlsx"):
        # convert to CSV first
        df = pd.read_excel(path)
        csv_path = path.replace(".xlsx", ".csv")
        df.to_csv(csv_path, index=False)
        return csv_path
    return path

def generate_image_block(prompt, filename):
    response = client.models.generate_content(
        model = "gemini-3.1-flash-image-preview",
        contents = prompt,
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data:
            with open(filename, "wb") as f:
                f.write(part.inline_data.data)

    return Block(
        type = BlockType.image_prompt,
        content = filename
    )

def generate_text_block(content):
    return Block(
        type = BlockType.text,
        content = content
    )

def generate_heading_block(heading):
    return Block(
        type = BlockType.heading,
        content = heading
    )

def generate_subheading_block(subheading):
    return Block(
        type = BlockType.subheading,
        content = subheading
    )

def request_clarification(question):
    return input(question)

tools = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="generate_image_block",
        description="Generate an image from a prompt and save it to a file, returns an image block",
        parameters={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Detailed image generation prompt"},
                "filename": {"type": "string", "description": "Unique output filename e.g. image_1.png"}
            },
            "required": ["prompt", "filename"]
        }
    ),
    types.FunctionDeclaration(
        name="generate_text_block",
        description="Generate a paragraph of body text for the document",
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The paragraph text content"}
            },
            "required": ["content"]
        }
    ),
    types.FunctionDeclaration(
        name="generate_heading_block",
        description="Generate a top-level heading for a new section",
        parameters={
            "type": "object",
            "properties": {
                "heading": {"type": "string", "description": "The heading text"}
            },
            "required": ["heading"]
        }
    ),
    types.FunctionDeclaration(
        name="generate_subheading_block",
        description="Generate a subheading within a section",
        parameters={
            "type": "object",
            "properties": {
                "subheading": {"type": "string", "description": "The subheading text"}
            },
            "required": ["subheading"]
        }
    ),
    types.FunctionDeclaration(
        name="request_clarification",
        description="Ask the user a clarifying question before proceeding, use sparingly",
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The clarifying question to ask the user"}
            },
            "required": ["question"]
        }
    ),
])

tool_map = {
    "generate_image_block": lambda args: generate_image_block(args["prompt"], args["filename"]),
    "generate_text_block": lambda args: generate_text_block(args["content"]),
    "generate_heading_block": lambda args: generate_heading_block(args["heading"]),
    "generate_subheading_block": lambda args: generate_subheading_block(args["subheading"]),
    "request_clarification": lambda args: request_clarification(args["question"]),
}

def build_executive_summary(file_paths):
    #Upload files to gemini api
    uploaded_files = []
    
    for path in file_paths:
        print(f"Uploading {path}...")
        path = prepare_file(path)
        file_ref = client.files.upload(file = path)

        while file_ref.state == "PROCESSING":
            time.sleep(2)
            file_ref = client.files.get(name = file_ref.name)

        uploaded_files.append(file_ref)
  

    #Agentically generate executive summary, intaking prompt and file names
    contents = [*uploaded_files, "Analyze all the files and generate a single cohesive executive summary. Use tools to build structure and ask for clarification when needed."]
    blocks = []

    while True:
        response = client.models.generate_content(
            model = "gemini-3.1-flash-lite-preview",
            contents = contents,
            config = types.GenerateContentConfig(
                tools = [tools],
            ),
        )

        tool_calls = [p for p in response.candidates[0].content.parts if p.function_call]
        if not tool_calls:
            #If not tool calls, model is done - return results
            title = response.text
            return Document(title = title, blocks = blocks)

        tool_results = []
        for part in tool_calls:
            fn_name = part.function_call.name
            fn_args = dict(part.function_call.args)
            result = tool_map[fn_name](fn_args)

            if isinstance(result, Block):
                blocks.append(result)

            tool_results.append(types.Part.from_function_response(
                name = fn_name,
                response = {"result": "Block created successfully"}
            ))
        
        contents.append(response.candidates[0].content)
        contents.append(types.Content(role = "tool", parts = tool_results))


#Main testing (Use python3 main.py + file names seperated by space)

if __name__ == "__main__":
    import sys

    test_files = sys.argv[1:] if len(sys.argv) > 1 else ["test.txt"]

    print(f"Building executive summary for: {test_files}")
    doc = build_executive_summary(test_files)

    print(f"\nTitle: {doc.title}")
    print(f"Blocks ({len(doc.blocks)}):")
    for i, block in enumerate(doc.blocks):
        print(f"  [{i}] {block.type.value}: {block.content[:80]}...")