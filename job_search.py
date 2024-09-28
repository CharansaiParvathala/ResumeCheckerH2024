
import requests
import json

# Set your Jooble API key here (you need to sign up at https://jooble.org/api/about)
API_KEY = "a9db5855-755b-43c3-9308-0e41c5702ba8"  # Replace this with your actual Jooble API key
JOOBLE_URL = "https://jooble.org/api/" + API_KEY

# Function to fetch job listings from Jooble API
def fetch_jobs_from_jooble(query, location="Remote"):
    headers = {
        'Content-Type': 'application/json',
    }
    payload = {
        'keywords': query,
        'location': location,
    }
    response = requests.post(JOOBLE_URL, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        return response.json().get('jobs', [])
    else:
        st.error(f"Failed to fetch data. Error Code: {response.status_code}")
        return []
