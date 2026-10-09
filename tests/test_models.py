from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

print("Available Gemini models:\n")

for model in client.models.list():
    print(model.name)