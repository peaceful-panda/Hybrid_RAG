import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

print("Sending request to Gemini...")

try:
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents="What does RAG stand for? Answer in one sentence."
    )

    print("\nGemini response:")
    print(response.text)

except Exception as error:
    print("\nGemini error:")
    print(type(error).__name__)
    print(error)