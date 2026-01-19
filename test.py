import requests

features = [1.08, 85.5, 103.2, 120.3, 119.8, 118.7, 120.0]
response = requests.post("http://127.0.0.1:8000/predict", json={"data": features})
print(response.json())
