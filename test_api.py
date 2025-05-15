import requests
import os
os.environ["TOGETHER_API_KEY"] = "414229540a05a7ce253fd2bfc33d221a19ff3277c6bd7bde1056eca4bc0f18ae"  # Removed the API Key that we used

url = 'https://api.together.xyz/inference'
headers = {
    'Authorization': f'Bearer {os.environ["TOGETHER_API_KEY"]}',
    'accept': 'application/json',
    'content-type': 'application/json'
}
data = {
    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "prompt": "Say hello",
    "max_tokens": 10,
    "temperature": 0.7
}

response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    print("API gọi thành công!")
    print("Phản hồi từ model:", response.json()['output']['choices'][0]['text'])
else:
    print(f"Lỗi khi gọi API: {response.status_code}")
    print("Thông điệp lỗi:", response.text)
