from fastapi import FastAPI, UploadFile, File, Form
from google import genai
from google.genai import types
import base64
import os

# Load environment variable
api_key = ("AIzaSyCfyvK6MhIyr4Dc0BrJg3T2C1N05EB1Wy0")
if not api_key:
    raise EnvironmentError("❌ GEMINI_API_KEY not set in environment variables!")

client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"

# Initialize FastAPI
app = FastAPI()

@app.post("/generate-description/")
async def generate_description(image: UploadFile = File(...), question: str = Form(...)):
    # Read the uploaded image
    image_data = await image.read()

    # Convert image to base64
    image_base64 = base64.b64encode(image_data).decode("utf-8")

    # Prepare the content for the model
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=base64.b64decode(image_base64),
                ),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text="""The image shows a screenshot of a phone displaying a college admit card. The card appears to belong to someone named "NIHARIKA". The phone's battery is at 92% and there's no internet connection."""
                ),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=question),
            ],
        ),
    ]

    # Set up content configuration
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""IF I AM BLIND, EXPLAIN ABOUT THE IMAGE IN 30 WORDS. Caption and answer questions about images""")
        ]
    )

    # Generate the content using the model
    response_text = ""
    
    # Use a regular for loop (no async) since it's a synchronous generator
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response_text += chunk.text

    return {"description": response_text}


# Run the FastAPI app using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

