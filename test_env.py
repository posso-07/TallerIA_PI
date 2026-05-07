from dotenv import load_dotenv
import os
from pathlib import Path

base = Path('DjangoProjectBase')
load_dotenv(base / 'key2_1.env')
load_dotenv(base / 'openAI.env')

api_key = os.getenv('openai_apikey')
print(f"API Key loaded: {api_key is not None}")
if api_key:
    print(f"First 20 chars: {api_key[:20]}")
    print(f"Length: {len(api_key)}")
else:
    print("ERROR: API Key is None")
