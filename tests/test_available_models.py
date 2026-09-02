"""
List all Groq models available to your API key.
Run with: py scripts/list_models.py
"""

import requests
from app.config import settings

resp = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {settings.groq_api_key}"},
    timeout=30,
)
resp.raise_for_status()

models = sorted(m["id"] for m in resp.json()["data"])

print(f"\n{len(models)} models available:\n")
for m in models:
    print(f"  - {m}")