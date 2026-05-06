import os

from dotenv import load_dotenv
from litellm import completion

load_dotenv()
model = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash")

print(f"Checking configured LiteLLM model: {model}")

try:
    response = completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: model check ok",
            }
        ],
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error calling configured model: {e}")
