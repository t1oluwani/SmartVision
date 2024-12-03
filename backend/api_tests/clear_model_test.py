import requests

response = requests.get(f"http://localhost:5000/clear/CNN")

if response.status_code == 200:
    print(response.json()["message"])


response = requests.get(f"http://localhost:5000/clear/FNN")

if response.status_code == 200:
    print(response.json()["message"])
    
response = requests.get(f"http://localhost:5000/clear/LR")

if response.status_code == 200:
    print(response.json()["message"])

