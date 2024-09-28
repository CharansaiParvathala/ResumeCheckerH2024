import streamlit as st
import requests
import json

# Set your Jooble API key here (you need to sign up at https://jooble.org/api/about)
API_KEY = "a9db5855-755b-43c3-9308-0e41c5702ba8"  # Replace this with your actual Jooble API key
JOOBLE_URL = "https://jooble.org/api/" + API_KEY

# Streamlit app configuration
st.title("Job Listings Fetcher using Jooble API")

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

# Dynamic Job Listings Section
st.subheader("Enter Job Search Criteria")

job_query = st.text_input("Job Title or Keywords", "Software Developer")
job_location = st.text_input("Job Location", "Remote")

# Fetch jobs when user clicks the button
if st.button("Search Jobs"):
    with st.spinner("Fetching job listings..."):
        job_listings = fetch_jobs_from_jooble(job_query, job_location)

        if job_listings:
            st.subheader(f"Job Listings for '{job_query}' in '{job_location}':")
            for job in job_listings:
                job_title = job.get('title', 'No title provided')
                job_company = job.get('company', 'No company provided')
                job_location = job.get('location', 'No location provided')
                job_salary = job.get('salary', 'Not mentioned')
                job_link = job.get('link', '#')

                st.write(f"{job_title}** at *{job_company}* - {job_location}")
                st.write(f"Salary: {job_salary}")
                st.write(f"Link: [Job Link]({job_link})")
                st.markdown("---")
        else:
            st.warning("No job listings found.")

# Optional manual refresh button
if st.button("Refresh Listings"):
    st.experimental_rerun()

# Display instructions
st.markdown("""
This app fetches job listings from the *Jooble API*.
Enter a job title and location to get the latest jobs.
""")
