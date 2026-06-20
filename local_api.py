import json

import requests

# Send a GET request to verify that the API is running.
r = requests.get("http://127.0.0.1:8000")

# Display the HTTP status code returned by the API.
print(f"Status Code: {r.status_code}")
# Display the welcome message returned from the API.
print(f"Result: {r.json()['message']}")



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST request containing Census Income data for prediction.
r = requests.post(
    "http://127.0.0.1:8000/data/",
    json=data,
)
# Display the HTTP status code returned by the prediction request.
print(f"Status Code: {r.status_code}")
# Display the predicted income classification.
print(f"Result: {r.json()['result']}")