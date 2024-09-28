common_job_titles = ['software developer', 'data scientist', 'project manager', 'machine learning engineer',
                     'data analyst', 'web developer', 'ui/ux designer', 'cloud engineer', 'devops engineer']

# Function to extract job title from the job description
def extract_job_title(job_description):
    # Convert the job description to lowercase for matching
    job_description_lower = job_description.lower()

    # Look for a common job title in the job description
    for title in common_job_titles:
        if title in job_description_lower:
            return title.title()  # Return the job title in title case
    return "Job Title Not Found"
