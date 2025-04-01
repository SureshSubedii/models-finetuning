import requests
import json

url = "http://localhost:4891/v1/chat/completions"

data = {
    "model": "Mistral Instruct",
    "messages": [{"role": "user", "content": "Who are you?"}],
    "max_tokens": 1000,
    "temperature": 0.28
}

response = requests.post(url, json=data)

if response.status_code == 200:
    json_data = response.json()
    print(json_data)
    print("___________________________________________________________________________________________")
    print("\n " + json_data["choices"][0]["message"]["content"])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
