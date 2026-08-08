from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai

print("Step 1: Loading environment...")

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

api_key = os.environ["GEMINI_API_KEY"]

print("✓ API Key Loaded")

client = genai.Client(api_key=api_key)

print("✓ Gemini Client Created")

text = """
Piezoelectricity is the coupling between mechanical
stress and electric polarization.
"""

print("Step 2: Requesting embedding...")

try:
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )

    print("✓ Embedding received!")

    print(response)

except Exception as e:
    print("ERROR:")
    print(type(e))
    print(e)
