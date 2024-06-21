import requests

url = "http://0.0.0.0:8000/predict"

# Test Case 1: Valid Input
response = requests.post(url, json={"text": "I am feeling very happy and excited today!"})
print("Test Case 1 - Valid Input")
print(response.json())

# Test Case 2: Missing Text Input
response = requests.post(url, json={})
print("\nTest Case 2 - Missing Text Input")
print(response.json())

# Test Case 3: No Matching Review Found
response = requests.post(url, json={"text": "This is some random text that probably doesn't exist in the dataset."})
print("\nTest Case 3 - No Matching Review Found")
print(response.json())
