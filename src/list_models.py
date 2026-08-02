from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai

# Load .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

for model in client.models.list():
    print(model.name)