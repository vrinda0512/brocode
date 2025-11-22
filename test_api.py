import requests
import json

BASE_URL = "http://127.0.0.1:5000"

print("🧪 Testing Watchlist API Endpoints...\n")

# Test 1: Get all watchlist items
print("Test 1: GET /api/watchlist")
response = requests.get(f"{BASE_URL}/api/watchlist")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\n")

# Test 2: Add to watchlist
print("Test 2: POST /api/watchlist/add")
new_item = {
    "type": "address",
    "value": "TEST123456789ABCDEF",
    "risk_level": "HIGH",
    "reason": "Test suspicious address",
    "tags": ["test", "suspicious"],
    "notes": "This is a test entry"
}
response = requests.post(f"{BASE_URL}/api/watchlist/add", json=new_item)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\n")

# Test 3: Check if on watchlist
print("Test 3: GET /api/watchlist/check/TEST123456789ABCDEF")
response = requests.get(f"{BASE_URL}/api/watchlist/check/TEST123456789ABCDEF")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\n")

# Test 4: Get watchlist stats
print("Test 4: GET /api/watchlist/stats")
response = requests.get(f"{BASE_URL}/api/watchlist/stats")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\n")

# Test 5: Custom prediction with watchlist check
print("Test 5: POST /api/predict_custom (with watchlisted address)")
transaction = {
    "amount": 50,
    "num_inputs": 5,
    "num_outputs": 3,
    "fee": 0.001,
    "sender_address": "TEST123456789ABCDEF",
    "receiver_address": "RECEIVER123"
}
response = requests.post(f"{BASE_URL}/api/predict_custom", json=transaction)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\n")

print("✅ API tests completed!")