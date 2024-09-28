
def generate_questions(job_description, job_skills):
    questions = []
    if "lead" in job_description.lower():
        questions.append("Can you describe your leadership style?")
    if "project management" in job_description.lower():
        questions.append("What project management methodologies are you familiar with?")
    if "python" in job_skills:
        questions.append("Can you explain how you would use Python for data analysis?")
    if "machine learning" in job_skills:
        questions.append("What machine learning frameworks have you worked with?")
    if "communication" in job_description.lower():
        questions.append("How do you handle communication in a team setting?")
    if "cloud computing" in job_description.lower():
        questions.append("What experience do you have with cloud platforms like AWS or Azure?")

    # Additional job roles and questions
    roles_questions = {
        'Software Engineer': [
            "How do you approach debugging a complex system?",
            "Can you describe a challenging project you've worked on?",
            "What programming languages are you most proficient in?",
            "Explain the concept of RESTful APIs."
        ],
        'Data Scientist': [
            "What is your experience with statistical modeling?",
            "How do you handle missing data in a dataset?",
            "What tools do you use for data visualization?",
            "Can you explain the difference between supervised and unsupervised learning?"
        ],
        'Project Manager': [
            "How do you prioritize tasks in a project?",
            "What methods do you use to communicate with your team?",
            "Can you describe a time when a project did not go as planned?",
            "How do you manage project risks?"
        ],
        'HR Manager': [
            "What strategies do you use for employee engagement?",
            "How do you handle conflicts between team members?",
            "Can you describe your experience with recruitment processes?",
            "What metrics do you use to evaluate HR effectiveness?"
        ],
        'DevOps Engineer': [
            "What is your experience with CI/CD pipelines?",
            "How do you ensure system security during deployments?",
            "Can you explain the concept of infrastructure as code?",
            "What tools do you use for monitoring applications?"
        ],
        'Business Analyst': [
            "How do you gather requirements from stakeholders?",
            "What methodologies do you use for analyzing business processes?",
            "Can you provide an example of a successful project you've managed?",
            "How do you handle changing requirements during a project?"
        ],
        'UX/UI Designer': [
            "How do you approach user research?",
            "What design tools do you use in your workflow?",
            "Can you describe your process for creating wireframes?",
            "How do you test your designs with users?"
        ],
        'Network Engineer': [
            "What experience do you have with network protocols?",
            "How do you troubleshoot network issues?",
            "Can you explain the concept of subnetting?",
            "What tools do you use for network monitoring?"
        ]
    }

    for role, qs in roles_questions.items():
        if role.lower() in job_description.lower():
            questions.extend(qs)

    return questions

# Updated Job role video links (YouTube videos accessible in India)
job_video_links = {
    'Software Engineer': 'https://www.youtube.com/watch?v=7Sg2Z26Fw08',
    'Data Scientist': 'https://www.youtube.com/watch?v=Yd1O4t5G8fI',
    'Project Manager': 'https://www.youtube.com/watch?v=5fLJ6d3vEwE',
    'HR Manager': 'https://www.youtube.com/watch?v=Jw8a0zZK1Y8',
    'DevOps Engineer': 'https://www.youtube.com/watch?v=COgH82fD1Cw',
    'Business Analyst': 'https://www.youtube.com/watch?v=RG7dj1jRO2Q',
    'UX/UI Designer': 'https://www.youtube.com/watch?v=HBg9s8h7sQg',
    'Network Engineer': 'https://www.youtube.com/watch?v=GZ0W9YI_3Nc'
}

def get_video_link(job_description):
    # Try to identify a matching role from the job description
    for role in job_video_links.keys():
        if role.lower() in job_description.lower():
            return [(role, job_video_links[role])]
    # Default return if no match
    return [("No role found", "No video available for this job role.")]
