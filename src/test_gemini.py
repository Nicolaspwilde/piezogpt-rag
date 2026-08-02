from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai

# ----------------------------
# Load Environment Variables
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

api_key = os.environ["GEMINI_API_KEY"]

# ----------------------------
# Create Gemini Client
# ----------------------------

client = genai.Client(api_key=api_key)

# ----------------------------
# Send Prompt
# ----------------------------

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Introduce yourself in two sentences."
)

print("=" * 60)
print(response.text)
print("=" * 60)