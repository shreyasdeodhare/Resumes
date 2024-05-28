# import json
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Define the job roles and skills
# job_roles = {
#     "Data Scientist": ["Python", "R", "SQL", "Machine Learning", "Data Analysis"],
#     "Software Engineer": ["Java", "JavaScript", "Python", "SQL", "HTML/CSS"],
#     "Full Stack Developer": ["JavaScript", "Node.js", "React", "Angular", "MongoDB"],
#     "Product Manager": ["Product Management", "Agile Methodologies", "UX/UI Design", "Market Research", "Project Management"],
#     "Business Analyst": ["Business Analysis", "Requirements Gathering", "Data Analysis", "SQL", "Project Management"],
#     "Digital Marketing Specialist": ["SEO", "SEM", "Social Media Marketing", "Google Analytics", "Content Marketing"],
#     "Sales Executive": ["Sales", "Customer Relationship Management (CRM)", "Negotiation", "Communication Skills", "Lead Generation"],
#     "Marketing Manager": ["Marketing Strategy", "Brand Management", "Market Research", "Digital Marketing", "Content Creation"],
#     "Quality Assurance Engineer": ["Manual Testing", "Automated Testing", "Test Planning", "Bug Tracking", "Agile Methodologies"],
#     "Network Engineer": ["Networking", "Cisco", "LAN/WAN", "Routing and Switching", "Network Security"],
#     "UI/UX Designer": ["User Interface Design", "User Experience Design", "Wireframing", "Prototyping", "Adobe Creative Suite"],
#     "DevOps Engineer": ["Linux", "Docker", "Kubernetes", "CI/CD", "Infrastructure as Code"],
#     "Cybersecurity Analyst": ["Cybersecurity", "Penetration Testing", "Firewall Management", "Incident Response", "Security Operations"],
#     "Financial Analyst": ["Financial Modeling", "Financial Reporting", "Data Analysis", "Excel", "Financial Planning"],
#     "Content Writer": ["Content Writing", "Copywriting", "SEO Writing", "Content Strategy", "Editing"],
#     "Human Resources Manager": ["Recruitment", "Employee Relations", "Performance Management", "HR Policies", "Training and Development"],
#     "Graphic Designer": ["Adobe Photoshop", "Illustrator", "InDesign", "Typography", "Visual Communication"],
#     "Project Manager": ["Project Management", "Risk Management", "Stakeholder Management", "Agile Methodologies", "Budgeting"],
#     "Operations Manager": ["Operations Management", "Process Improvement", "Supply Chain Management", "Logistics", "Inventory Management"],
#     "Software Architect": ["Software Architecture", "System Design", "Microservices", "Cloud Computing", "Design Patterns"],
#     "Data Engineer": ["Python", "SQL", "Big Data", "Hadoop", "Spark", "Kafka", "ETL", "Data Warehousing", "Airflow", "NoSQL", "AWS", "Azure", "Google Cloud Platform"],
#     "Technical Support Engineer": ["Troubleshooting", "Customer Support", "Networking", "Operating Systems", "Ticketing Systems"],
#     "UX Researcher": ["User Research", "Usability Testing", "Qualitative Research", "Quantitative Research", "Psychology"],
#     "Mobile App Developer": ["Android", "iOS", "Swift", "Java", "Mobile UI Design"],
#     "Cloud Solutions Architect": ["AWS", "Azure", "Google Cloud Platform", "Serverless Architecture", "DevOps"],
#     "Supply Chain Manager": ["Supply Chain Optimization", "Inventory Management", "Procurement", "Logistics", "Supplier Relationship Management"],
#     "Java Full Stack Developer": ["Core Java", "Spring", "MySQL", "Salesforce", "Spring MVC", "Spring JDBC", "HTML CSS", "Eclipse IDE", "VS Code", "Azure Fundamentals"]
# }

# # Function to clean and beautify JSON data
# def clean_and_beautify_json(json_data):
#     data = json.loads(json_data)
    
#     # Clean and validate each field
#     for key, value in data.items():
#         if isinstance(value, str):
#             data[key] = value.strip()
#         elif isinstance(value, list):
#             data[key] = [item.strip() if isinstance(item, str) else item for item in value]
#         elif isinstance(value, dict):
#             data[key] = {sub_key: sub_value.strip() if isinstance(sub_value, str) else sub_value for sub_key, sub_value in value.items()}

#     # Beautify the JSON data
#     beautified_json_data = json.dumps(data, indent=4)
#     return data, beautified_json_data

# # Text Preprocessing
# def preprocess_text(text):
#     # Convert to lowercase
#     text = text.lower()
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     return ' '.join(tokens)

# # Determine the best job role
# def find_best_role(candidate_data):
#     # Extract candidate skills
#     candidate_skills = candidate_data.get('skills', "")
    
#     # Extract job role and responsibilities from previous company information
#     job_responsibilities = ""
#     if 'previousCompanyInformation' in candidate_data:
#         for company_info in candidate_data['previousCompanyInformation']:
#             if 'jobRoleAndResponsibilities' in company_info and company_info['jobRoleAndResponsibilities']:
#                 job_responsibilities += company_info['jobRoleAndResponsibilities'] + " "
    
#     # Preprocess candidate skills and job responsibilities
#     candidate_skills_preprocessed = preprocess_text(candidate_skills)
#     job_responsibilities_preprocessed = preprocess_text(job_responsibilities)
    
#     # Combine job role skills into a single string
#     job_roles_combined = {role: preprocess_text(' '.join(skills)) for role, skills in job_roles.items()}
    
#     # Initialize TF-IDF Vectorizer
#     tfidf_vectorizer = TfidfVectorizer()
    
#     # Fit the vectorizer on the candidate skills and job role skills
#     tfidf_matrix = tfidf_vectorizer.fit_transform([candidate_skills_preprocessed] + [job_responsibilities_preprocessed] + list(job_roles_combined.values()))
    
#     # Calculate cosine similarity
#     cosine_similarities = cosine_similarity(tfidf_matrix[0:2], tfidf_matrix[2:]).flatten()
    
#     # Check if there are valid similarities
#     if len(cosine_similarities) == 0:
#         return "No suitable job role found"
    
#     # Find the index of the highest similarity score
#     highest_score_index = cosine_similarities.argmax()
    
#     # Find the corresponding job role
#     best_role = list(job_roles.keys())[highest_score_index]
    
#     return best_role

# # Process the candidate data and find the best job role
# def process_candidate_data(json_data):
#     data, beautified_json_data = clean_and_beautify_json(json_data)
    
#     # Find the best job role
#     best_role = find_best_role(data)

#     print("Beautified JSON Data:")
#     print(beautified_json_data)
#     print("\nBest Job Role for the Candidate:")
#     print(f"Role: {best_role}")

# # JSON Data
# json_data = '''
# {
#   "address": "Jatwal, Samba, J&K, 184141",
#   "email": "pks189137@gmail.com",
#   "skills": "Core Java,Spring,MySQL,Salesforce,Spring MVC,Spring JDBC,HTML CSS,Eclipse IDE,VS Code,Azure Fundamentals",
#   "city": "DELHI NCR, PUNE",
#   "certification": "Microsoft Azure Fundamental AZ-900",
#   "candidateName": "PARDEEP KUMAR",
#   "linkedUrl": "https://www.linkedin.com/in/pardeepgr/",
#   "phoneNumber": "9682189343",
#   "higherQualification": "B. TECH. COMPUTER SCIENCE ENGINEERING",
#   "passingYear": "2020",
#   "previousCompanyInformation": [
#     {
#       "Company_Name": "Tech Mahindra",
#       "start_date": "07/2021",
#       "end_date": "10/2022",
#       "Duration": null,
#       "jobRoleAndResponsibilities": "Developed software code as per specifications, by understanding customer requirements. Build Apps with Teammates and provided best solution. Worked on Software like Salesforce to build apps and automate various functionalities.",
#       "Project_Name": null
#     }
#   ],
#   "lastCompany": "Tech Mahindra"
# }
# '''

# # Process candidate data and find the best job role
# process_candidate_data(json_data)




import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define the job roles and skills
job_roles = {
    "Data Scientist": ["Python", "R", "SQL", "Machine Learning", "Data Analysis"],
    "Software Engineer": ["Java", "JavaScript", "Python", "SQL", "HTML/CSS"],
    "Full Stack Developer": ["JavaScript", "Node.js", "React", "Angular", "MongoDB"],
    "Product Manager": ["Product Management", "Agile Methodologies", "UX/UI Design", "Market Research", "Project Management"],
    "Business Analyst": ["Business Analysis", "Requirements Gathering", "Data Analysis", "SQL", "Project Management"],
    "Digital Marketing Specialist": ["SEO", "SEM", "Social Media Marketing", "Google Analytics", "Content Marketing"],
    "Sales Executive": ["Sales", "Customer Relationship Management (CRM)", "Negotiation", "Communication Skills", "Lead Generation"],
    "Marketing Manager": ["Marketing Strategy", "Brand Management", "Market Research", "Digital Marketing", "Content Creation"],
    "Quality Assurance Engineer": ["Manual Testing", "Automated Testing", "Test Planning", "Bug Tracking", "Agile Methodologies"],
    "Network Engineer": ["Networking", "Cisco", "LAN/WAN", "Routing and Switching", "Network Security"],
    "UI/UX Designer": ["User Interface Design", "User Experience Design", "Wireframing", "Prototyping", "Adobe Creative Suite"],
    "DevOps Engineer": ["Linux", "Docker", "Kubernetes", "CI/CD", "Infrastructure as Code"],
    "Cybersecurity Analyst": ["Cybersecurity", "Penetration Testing", "Firewall Management", "Incident Response", "Security Operations"],
    "Financial Analyst": ["Financial Modeling", "Financial Reporting", "Data Analysis", "Excel", "Financial Planning"],
    "Content Writer": ["Content Writing", "Copywriting", "SEO Writing", "Content Strategy", "Editing"],
    "Human Resources Manager": ["Recruitment", "Employee Relations", "Performance Management", "HR Policies", "Training and Development"],
    "Graphic Designer": ["Adobe Photoshop", "Illustrator", "InDesign", "Typography", "Visual Communication"],
    "Project Manager": ["Project Management", "Risk Management", "Stakeholder Management", "Agile Methodologies", "Budgeting"],
    "Operations Manager": ["Operations Management", "Process Improvement", "Supply Chain Management", "Logistics", "Inventory Management"],
    "Software Architect": ["Software Architecture", "System Design", "Microservices", "Cloud Computing", "Design Patterns"],
    "Data Engineer": ["Python", "SQL", "Big Data", "Hadoop", "Spark", "Kafka", "ETL", "Data Warehousing", "Airflow", "NoSQL", "AWS", "Azure", "Google Cloud Platform"],
    "Technical Support Engineer": ["Troubleshooting", "Customer Support", "Networking", "Operating Systems", "Ticketing Systems"],
    "UX Researcher": ["User Research", "Usability Testing", "Qualitative Research", "Quantitative Research", "Psychology"],
    "Mobile App Developer": ["Android", "iOS", "Swift", "Java", "Mobile UI Design"],
    "Cloud Solutions Architect": ["AWS", "Azure", "Google Cloud Platform", "Serverless Architecture", "DevOps"],
    "Supply Chain Manager": ["Supply Chain Optimization", "Inventory Management", "Procurement", "Logistics", "Supplier Relationship Management"],
    "Java Full Stack Developer": ["Core Java", "Spring", "MySQL", "Salesforce", "Spring MVC", "Spring JDBC", "HTML CSS", "Eclipse IDE", "VS Code", "Azure Fundamentals"],
    "Data Scientist": ["Python", "R", "SQL", "Machine Learning", "Data Analysis"],
    "Software Engineer": ["Java", "JavaScript", "Python", "SQL", "HTML/CSS"],
    "Full Stack Developer": ["JavaScript", "Node.js", "React", "Angular", "MongoDB"],
    "Product Manager": ["Product Management", "Agile Methodologies", "UX/UI Design", "Market Research", "Project Management"],
    "Business Analyst": ["Business Analysis", "Requirements Gathering", "Data Analysis", "SQL", "Project Management"],
    "Digital Marketing Specialist": ["SEO", "SEM", "Social Media Marketing", "Google Analytics", "Content Marketing"],
    "Sales Executive": ["Sales", "Customer Relationship Management (CRM)", "Negotiation", "Communication Skills", "Lead Generation"],
    "Marketing Manager": ["Marketing Strategy", "Brand Management", "Market Research", "Digital Marketing", "Content Creation"],
    "Quality Assurance Engineer": ["Manual Testing", "Automated Testing", "Test Planning", "Bug Tracking", "Agile Methodologies"],
    "Network Engineer": ["Networking", "Cisco", "LAN/WAN", "Routing and Switching", "Network Security"],
    "UI/UX Designer": ["User Interface Design", "User Experience Design", "Wireframing", "Prototyping", "Adobe Creative Suite"],
    "DevOps Engineer": ["Linux", "Docker", "Kubernetes", "CI/CD", "Infrastructure as Code"],
    "Cybersecurity Analyst": ["Cybersecurity", "Penetration Testing", "Firewall Management", "Incident Response", "Security Operations"],
    "Financial Analyst": ["Financial Modeling", "Financial Reporting", "Data Analysis", "Excel", "Financial Planning"],
    "Content Writer": ["Content Writing", "Copywriting", "SEO Writing", "Content Strategy", "Editing"],
    "Human Resources Manager": ["Recruitment", "Employee Relations", "Performance Management", "HR Policies", "Training and Development"],
    "Graphic Designer": ["Adobe Photoshop", "Illustrator", "InDesign", "Typography", "Visual Communication"],
    "Project Manager": ["Project Management", "Risk Management", "Stakeholder Management", "Agile Methodologies", "Budgeting"],
    "Operations Manager": ["Operations Management", "Process Improvement", "Supply Chain Management", "Logistics", "Inventory Management"],
    "Software Architect": ["Software Architecture", "System Design", "Microservices", "Cloud Computing", "Design Patterns"],
    "Data Engineer": ["Python", "SQL", "Big Data", "Hadoop", "Spark", "Kafka", "ETL", "Data Warehousing", "Airflow", "NoSQL", "AWS", "Azure", "Google Cloud Platform"],
    "Technical Support Engineer": ["Troubleshooting", "Customer Support", "Networking", "Operating Systems", "Ticketing Systems"],
    "UX Researcher": ["User Research", "Usability Testing", "Qualitative Research", "Quantitative Research", "Psychology"],
    "Mobile App Developer": ["Android", "iOS", "Swift", "Java", "Mobile UI Design"],
    "Cloud Solutions Architect": ["AWS", "Azure", "Google Cloud Platform", "Serverless Architecture", "DevOps"],
    "Supply Chain Manager": ["Supply Chain Optimization", "Inventory Management", "Procurement", "Logistics", "Supplier Relationship Management"],
    "Data Scientist": ["Python", "R", "SQL", "Machine Learning", "Data Analysis"],
    "Software Engineer": ["Java", "JavaScript", "Python", "SQL", "HTML/CSS"],
    "Full Stack Developer": ["JavaScript", "Node.js", "React", "Angular", "MongoDB"],
    "Product Manager": ["Product Management", "Agile Methodologies", "UX/UI Design", "Market Research", "Project Management"],
    "Business Analyst": ["Business Analysis", "Requirements Gathering", "Data Analysis", "SQL", "Project Management"],
    "Digital Marketing Specialist": ["SEO", "SEM", "Social Media Marketing", "Google Analytics", "Content Marketing"],
    "Sales Executive": ["Sales", "Customer Relationship Management (CRM)", "Negotiation", "Communication Skills", "Lead Generation"],
    "Marketing Manager": ["Marketing Strategy", "Brand Management", "Market Research", "Digital Marketing", "Content Creation"],
    "Quality Assurance Engineer": ["Manual Testing", "Automated Testing", "Test Planning", "Bug Tracking", "Agile Methodologies"],
    "Network Engineer": ["Networking", "Cisco", "LAN/WAN", "Routing and Switching", "Network Security"],
    "UI/UX Designer": ["User Interface Design", "User Experience Design", "Wireframing", "Prototyping", "Adobe Creative Suite"],
    "DevOps Engineer": ["Linux", "Docker", "Kubernetes", "CI/CD", "Infrastructure as Code"],
    "Cybersecurity Analyst": ["Cybersecurity", "Penetration Testing", "Firewall Management", "Incident Response", "Security Operations"],
    "Financial Analyst": ["Financial Modeling", "Financial Reporting", "Data Analysis", "Excel", "Financial Planning"],
    "Content Writer": ["Content Writing", "Copywriting", "SEO Writing", "Content Strategy", "Editing"],
    "Human Resources Manager": ["Recruitment", "Employee Relations", "Performance Management", "HR Policies", "Training and Development"],
    "Graphic Designer": ["Adobe Photoshop", "Illustrator", "InDesign", "Typography", "Visual Communication"],
    "Project Manager": ["Project Management", "Risk Management", "Stakeholder Management", "Agile Methodologies", "Budgeting"],
    "Operations Manager": ["Operations Management", "Process Improvement", "Supply Chain Management", "Logistics", "Inventory Management"],
    "Software Architect": ["Software Architecture", "System Design", "Microservices", "Cloud Computing", "Design Patterns"],
    "Data Engineer": ["Big Data", "Hadoop", "Spark", "ETL", "Data Warehousing"],
    "Technical Support Engineer": ["Troubleshooting", "Customer Support", "Networking", "Operating Systems", "Ticketing Systems"],
    "UX Researcher": ["User Research", "Usability Testing", "Qualitative Research", "Quantitative Research", "Psychology"],
    "Mobile App Developer": ["Android", "iOS", "Swift", "Java", "Mobile UI Design"],
    "Cloud Solutions Architect": ["AWS", "Azure", "Google Cloud Platform", "Serverless Architecture", "DevOps"],
    "Supply Chain Manager": ["Supply Chain Optimization", "Inventory Management", "Procurement", "Logistics", "Supplier Relationship Management"],
    "Data Analyst": ["Data Analysis", "SQL", "Excel", "Statistics", "Data Visualization"],
    "IT Manager": ["IT Infrastructure", "IT Strategy", "Vendor Management", "Budgeting", "IT Security"],
    "Technical Writer": ["Technical Writing", "Documentation", "API Documentation", "User Manuals", "Content Management Systems"],
    "ERP Consultant": ["ERP Implementation", "SAP", "Oracle", "Business Process Mapping", "Change Management"],
    "Machine Learning Engineer": ["Machine Learning", "Deep Learning", "Neural Networks", "Python", "TensorFlow"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "React", "Front-end Frameworks"],
    "Systems Administrator": ["System Administration", "Networking", "Linux", "Windows Server", "Virtualization"],
    "AI Engineer": ["Artificial Intelligence", "Natural Language Processing", "Computer Vision", "Deep Learning", "Python"],
    "Database Administrator": ["Database Management", "SQL", "Database Optimization", "Backup and Recovery", "Performance Tuning"],
    "Product Owner": ["Product Ownership", "Scrum", "Agile Methodologies", "Backlog Management", "User Stories"],
    "UX/UI Designer": ["User Experience Design", "User Interface Design", "Wireframing", "Prototyping", "Adobe XD"],
    "Technical Recruiter": ["Recruitment", "Candidate Sourcing", "Interviewing", "Talent Acquisition", "HR"],
    "System Analyst": ["System Analysis", "Requirements Gathering", "Software Development Life Cycle (SDLC)", "Documentation", "UML"],
    "Healthcare Administrator": ["Healthcare Management", "Regulatory Compliance", "Medical Billing", "Electronic Health Records (EHR)", "Healthcare IT"],
    "Data Entry Operator": ["Data Entry", "Typing", "Microsoft Office", "Data Verification", "Data Processing"],
    "IT Security Specialist": ["Cybersecurity", "Information Security", "Penetration Testing", "Firewall Management", "Security Policies"],
    "Solution Architect": ["Solution Architecture", "Enterprise Architecture", "Technical Architecture", "Integration", "Cloud Computing"],
    "UI Developer": ["HTML", "CSS", "JavaScript", "Angular", "Vue.js"],
    "Business Development Manager": ["Business Development", "Sales", "Client Relationship Management", "Market Research", "Negotiation"],
    "Customer Success Manager": ["Customer Relationship Management (CRM)", "Customer Support", "Account Management", "Customer Satisfaction", "Retention Strategies"],
    "Data Entry Clerk": ["Data Entry", "Microsoft Excel", "Typing", "Data Verification", "Data Processing"],
    "System Administrator": ["System Administration", "Networking", "Linux", "Windows Server", "Virtualization"],
    "Healthcare Consultant": ["Healthcare Consulting", "Healthcare Management", "Healthcare IT", "Regulatory Compliance", "HIPAA"],
    "Network Administrator": ["Network Administration", "Cisco", "LAN/WAN", "Firewall Management", "Network Security"],
    "Sales Manager": ["Sales Management", "Business Development", "Sales Strategy", "Sales Operations", "Team Leadership"],
    "Business Intelligence Analyst": ["Business Intelligence", "Data Analysis", "SQL", "Data Visualization", "Dashboard Design"],
    "Technical Consultant": ["Technical Consulting", "Solution Architecture", "Client Management", "Technical Support", "Software Implementation"],
    "Technical Project Manager": ["Project Management", "Technical Leadership", "Agile Methodologies", "Stakeholder Management", "Risk Management"],
    "IT Consultant": ["IT Consulting", "Client Management", "Project Management", "Business Analysis", "Solution Architecture"],
    "Financial Controller": ["Financial Reporting", "Financial Analysis", "Budgeting", "Financial Planning", "Audit"],
    "Sales Representative": ["Sales", "Prospecting", "Customer Relationship Management (CRM)", "Negotiation", "Closing"],
    "Customer Service Representative": ["Customer Service", "Communication Skills", "Problem-Solving", "Multitasking", "Empathy"],
    "Recruiter": ["Recruitment", "Candidate Sourcing", "Interviewing", "Talent Acquisition", "HR"],
    "Accountant": ["Accounting", "Bookkeeping", "Financial Reporting", "Taxation", "Microsoft Excel"],
    "Marketing Coordinator": ["Marketing", "Event Planning", "Content Creation", "Social Media Management", "Email Marketing"],
    "Business Development Executive": ["Business Development", "Sales", "Lead Generation", "Client Relationship Management", "Communication Skills"],
    "Customer Support Specialist": ["Customer Support", "Technical Support", "Troubleshooting", "Customer Satisfaction", "Ticketing Systems"],
    "HR Generalist": ["Human Resources", "Recruitment", "Employee Relations", "Performance Management", "Training and Development"],
    "Project Coordinator": ["Project Management", "Project Planning", "Documentation", "Team Coordination", "Stakeholder Management"],
    "Legal Counsel": ["Legal Research", "Contract Negotiation", "Litigation Management", "Legal Writing", "Regulatory Compliance"],
    "Business Analyst Manager": ["Business Analysis", "Project Management", "Team Leadership", "Stakeholder Management", "Requirements Gathering"],
    "Sales Consultant": ["Sales", "Client Consultation", "Product Knowledge", "Negotiation", "Customer Relationship Management (CRM)"],
    "Financial Advisor": ["Financial Planning", "Investment Management", "Wealth Management", "Retirement Planning", "Insurance"],
    "Executive Assistant": ["Administrative Support", "Calendar Management", "Travel Arrangements", "Meeting Coordination", "Communication Skills"],
    "Brand Manager": ["Brand Management", "Marketing Strategy", "Brand Development", "Market Research", "Campaign Management"],
    "Account Manager": ["Account Management", "Client Relationship Management", "Sales", "Customer Support", "Negotiation"],
    "Social Media Manager": ["Social Media Management", "Content Creation", "Community Engagement", "Analytics", "Social Media Advertising"],
    "Administrative Assistant": ["Administrative Support", "Data Entry", "Calendar Management", "Microsoft Office", "Organizational Skills"],
    "Operations Coordinator": ["Operations Management", "Project Coordination", "Logistics", "Inventory Management", "Supply Chain"],
    "HR Coordinator": ["Human Resources", "Recruitment", "Onboarding", "Employee Relations", "Administrative Support"],
    "Legal Assistant": ["Legal Research", "Document Drafting", "Legal Administration", "Client Communication", "Filing"],
    "IT Support Specialist": ["Technical Support", "Troubleshooting", "Network Administration", "Hardware Support", "Software Installation"],
    "Marketing Assistant": ["Marketing Support", "Content Creation", "Social Media Management", "Email Marketing", "Event Coordination"],
    "Customer Service Manager": ["Customer Service", "Team Management", "Problem-Solving", "Customer Satisfaction", "Quality Assurance"],
    "Financial Analyst": ["Financial Analysis", "Financial Modeling", "Data Analysis", "Excel", "Budgeting"],
    "Operations Analyst": ["Operations Analysis", "Data Analysis", "Process Improvement", "Report Generation", "Performance Monitoring"],
    "Legal Secretary": ["Legal Assistance", "Document Preparation", "Client Communication", "Administrative Support", "Legal Research"],
    "IT Technician": ["Technical Support", "Hardware Installation", "Software Installation", "Troubleshooting", "Network Configuration"],
    "Market Research Analyst": ["Market Research", "Data Analysis", "Survey Design", "Statistical Analysis", "Report Writing"],
    "Accounting Clerk": ["Accounting", "Bookkeeping", "Invoicing", "Financial Records", "Microsoft Excel"],
    "IT Administrator": ["System Administration", "Network Administration", "Server Management", "Security Management", "User Support"],
    "Logistics Coordinator": ["Logistics Management", "Inventory Management", "Supply Chain Coordination", "Transportation Management", "Shipping"],
    "Legal Consultant": ["Legal Consulting", "Legal Research", "Contract Review", "Legal Compliance", "Legal Writing"],
    "Marketing Specialist": ["Marketing Strategy", "Campaign Management", "Content Creation", "Digital Marketing", "Email Marketing"],
    "Customer Service Coordinator": ["Customer Service", "Complaint Handling", "Order Processing", "Data Entry", "Call Handling"],
    "Financial Planner": ["Financial Planning", "Investment Management", "Retirement Planning", "Wealth Management", "Tax Planning"],
    "Operations Assistant": ["Operations Support", "Administrative Assistance", "Project Coordination", "Data Entry", "Documentation"],
    "IT Analyst": ["IT Analysis", "System Analysis", "Data Analysis", "Software Testing", "Troubleshooting"],
    "Legal Analyst": ["Legal Research", "Legal Documentation", "Case Analysis", "Regulatory Compliance", "Legal Writing"],
    "Marketing Analyst": ["Marketing Analysis", "Data Analysis", "Market Research", "Campaign Performance Analysis", "Statistical Analysis"],
    "Customer Support Representative": ["Customer Support", "Technical Support", "Troubleshooting", "Ticketing Systems", "Communication Skills"],
    "Accounting Manager": ["Accounting", "Financial Reporting", "Budgeting", "Financial Analysis", "Team Management"],
    "IT Specialist": ["IT Support", "Hardware Maintenance", "Software Installation", "Network Administration", "Troubleshooting"],
    "Logistics Manager": ["Logistics Management", "Supply Chain Management", "Inventory Management", "Transportation Management", "Warehousing"],
    "Legal Advisor": ["Legal Advice", "Legal Research", "Document Drafting", "Client Consultation", "Litigation Support"],
    "Marketing Coordinator": ["Marketing Support", "Event Coordination", "Content Creation", "Social Media Management", "Email Marketing"],
    "Customer Service Supervisor": ["Customer Service", "Team Management", "Performance Monitoring", "Training", "Quality Assurance"],
    "Financial Controller": ["Financial Reporting", "Financial Analysis", "Budgeting", "Forecasting", "Financial Strategy"],
    "Operations Supervisor": ["Operations Management", "Team Leadership", "Process Improvement", "Workflow Management", "Quality Control"],
    "Legal Executive": ["Legal Documentation", "Document Drafting", "Contract Review", "Legal Research", "Legal Compliance"],
    "Marketing Executive": ["Marketing", "Campaign Management", "Content Creation", "Digital Marketing", "Social Media Management"],
    "Customer Service Associate": ["Customer Support", "Issue Resolution", "Order Processing", "Complaint Handling", "Call Handling"],
    "Financial Accountant": ["Financial Accounting", "Financial Reporting", "Taxation", "Audit", "Financial Analysis"],
    "Operations Administrator": ["Operations Support", "Administrative Assistance", "Project Coordination", "Data Entry", "Documentation"],
    "Legal Manager": ["Legal Management", "Team Leadership", "Legal Strategy", "Compliance Management", "Litigation Management"],
    "Marketing Director": ["Marketing Strategy", "Campaign Management", "Brand Development", "Digital Marketing", "Market Research"],
    "Customer Service Assistant": ["Customer Service", "Administrative Support", "Data Entry", "Order Processing", "Call Handling"],
    "Financial Analyst": ["Financial Analysis", "Financial Modeling", "Budgeting", "Forecasting", "Data Analysis"],
    "Operations Analyst": ["Operations Analysis", "Process Improvement", "Data Analysis", "Performance Monitoring", "Report Generation"],
    "Legal Secretary": ["Legal Assistance", "Document Preparation", "Client Communication", "Administrative Support", "Legal Research"],
    "Marketing Assistant": ["Marketing Support", "Content Creation", "Social Media Management", "Email Marketing", "Event Coordination"],
    "Customer Service Manager": ["Customer Service", "Team Management", "Problem-Solving", "Customer Satisfaction", "Quality Assurance"],
    "Financial Analyst": ["Financial Analysis", "Financial Modeling", "Data Analysis", "Excel", "Budgeting"],
    "Operations Analyst": ["Operations Analysis", "Data Analysis", "Process Improvement", "Report Generation", "Performance Monitoring"],
    "Legal Secretary": ["Legal Assistance", "Document Preparation", "Client Communication", "Administrative Support", "Legal Research"],
    "IT Technician": ["Technical Support", "Hardware Installation", "Software Installation", "Troubleshooting", "Network Configuration"],
    "Market Research Analyst": ["Market Research", "Data Analysis", "Survey Design", "Statistical Analysis", "Report Writing"],
    "Accounting Clerk": ["Accounting", "Bookkeeping", "Invoicing", "Financial Records", "Microsoft Excel"],
    "IT Administrator": ["System Administration", "Network Administration", "Server Management", "Security Management", "User Support"],
    "Logistics Coordinator": ["Logistics Management", "Inventory Management", "Supply Chain Coordination", "Transportation Management", "Shipping"],
    "Legal Consultant": ["Legal Consulting", "Legal Research", "Contract Review", "Legal Compliance", "Legal Writing"],
    "Marketing Specialist": ["Marketing Strategy", "Campaign Management", "Content Creation", "Digital Marketing", "Email Marketing"],
    "Customer Service Coordinator": ["Customer Service", "Complaint Handling", "Order Processing", "Data Entry", "Call Handling"],
    "Financial Planner": ["Financial Planning", "Investment Management", "Retirement Planning", "Wealth Management", "Tax Planning"],
    "Operations Assistant": ["Operations Support", "Administrative Assistance", "Project Coordination", "Data Entry", "Documentation"],
    "IT Analyst": ["IT Analysis", "System Analysis", "Data Analysis", "Software Testing", "Troubleshooting"],
    "Legal Analyst": ["Legal Research", "Legal Documentation", "Case Analysis", "Regulatory Compliance", "Legal Writing"],
    "Marketing Analyst": ["Marketing Analysis", "Data Analysis", "Market Research", "Campaign Performance Analysis", "Statistical Analysis"],
    "Customer Support Representative": ["Customer Support", "Technical Support", "Troubleshooting", "Ticketing Systems", "Communication Skills"],
    "Accounting Manager": ["Accounting", "Financial Reporting", "Budgeting", "Financial Analysis", "Team Management"],
    "IT Specialist": ["IT Support", "Hardware Maintenance", "Software Installation", "Network Administration", "Troubleshooting"],
    "Logistics Manager": ["Logistics Management", "Supply Chain Management", "Inventory Management", "Transportation Management", "Warehousing"],
    "Legal Advisor": ["Legal Advice", "Legal Research", "Document Drafting", "Client Consultation", "Litigation Support"],
    "Marketing Coordinator": ["Marketing Support", "Event Coordination", "Content Creation", "Social Media Management", "Email Marketing"],
    "Customer Service Supervisor": ["Customer Service", "Team Management", "Performance Monitoring", "Training", "Quality Assurance"],
    "Financial Controller": ["Financial Reporting", "Financial Analysis", "Budgeting", "Forecasting", "Financial Strategy"],
    "Operations Supervisor": ["Operations Management", "Team Leadership", "Process Improvement", "Workflow Management", "Quality Control"],
    "Legal Executive": ["Legal Documentation", "Document Drafting", "Contract Review", "Legal Research", "Legal Compliance"],
    "Marketing Executive": ["Marketing", "Campaign Management", "Content Creation", "Digital Marketing", "Social Media Management"],
    "Customer Service Associate": ["Customer Support", "Issue Resolution", "Order Processing", "Complaint Handling", "Call Handling"],
    "Financial Accountant": ["Financial Accounting", "Financial Reporting", "Taxation", "Audit", "Financial Analysis"],
    "Operations Administrator": ["Operations Support", "Administrative Assistance", "Project Coordination", "Data Entry", "Documentation"],
    "Legal Manager": ["Legal Management", "Team Leadership", "Legal Strategy", "Compliance Management", "Litigation Management"],
    "Marketing Director": ["Marketing Strategy", "Campaign Management", "Brand Development", "Digital Marketing", "Market Research"],
    "Customer Service Assistant": ["Customer Service", "Administrative Support", "Data Entry", "Order Processing", "Call Handling"],
    "Financial Analyst": ["Financial Analysis", "Financial Modeling", "Budgeting", "Forecasting", "Data Analysis"],
    "Operations Analyst": ["Operations Analysis", "Process Improvement", "Data Analysis", "Performance Monitoring", "Report Generation"],
    "Legal Secretary": ["Legal Assistance", "Document Preparation", "Client Communication", "Administrative Support", "Legal Research"],
    "Marketing Assistant": ["Marketing Support", "Content Creation", "Social Media Management", "Email Marketing", "Event Coordination"],
    "Customer Service Manager": ["Customer Service", "Team Management", "Problem-Solving", "Customer Satisfaction", "Quality Assurance"],
    "Financial Analyst": ["Financial Analysis", "Financial Modeling", "Data Analysis", "Excel", "Budgeting"],
    "Operations Analyst": ["Operations Analysis", "Data Analysis", "Process Improvement", "Report Generation", "Performance Monitoring"],
    "Legal Secretary": ["Legal Assistance", "Document Preparation", "Client Communication", "Administrative Support", "Legal Research"],
    "IT Technician": ["Technical Support", "Hardware Installation", "Software Installation", "Troubleshooting", "Network Configuration"],
    "Market Research Analyst": ["Market Research", "Data Analysis", "Survey Design", "Statistical Analysis", "Report Writing"],
    "Accounting Clerk": ["Accounting", "Bookkeeping", "Invoicing", "Financial Records", "Microsoft Excel"],
    "IT Administrator": ["System Administration", "Network Administration", "Server Management", "Security Management", "User Support"],
    "Logistics Coordinator": ["Logistics Management", "Inventory Management", "Supply Chain Coordination", "Transportation Management", "Shipping"],
    "Legal Consultant": ["Legal Consulting", "Legal Research", "Contract Review", "Legal Compliance", "Legal Writing"],
    "Marketing Specialist": ["Marketing Strategy", "Campaign Management", "Content Creation", "Digital Marketing", "Email Marketing"],
    "Customer Service Coordinator": ["Customer Service", "Complaint Handling", "Order Processing", "Data Entry", "Call Handling"],
    "Financial Planner": ["Financial Planning", "Investment Management", "Retirement Planning", "Wealth Management", "Tax Planning"],
    "Operations Assistant": ["Operations Support", "Administrative Assistance", "Project Coordination", "Data Entry", "Documentation"],
    "IT Analyst": ["IT Analysis", "System Analysis", "Data Analysis", "Software Testing", "Troubleshooting"],
    "Legal Analyst": ["Legal Research", "Legal Documentation", "Case Analysis", "Regulatory Compliance", "Legal Writing"],
    "Marketing Analyst": ["Marketing Analysis", "Data Analysis", "Market Research", "Campaign Performance Analysis", "Statistical Analysis"],
    "Customer Support Representative": ["Customer Support", "Technical Support", "Troubleshooting", "Ticketing Systems", "Communication Skills"],
    "Accounting Manager": ["Accounting", "Financial Reporting", "Budgeting", "Financial Analysis", "Team Management"],
    "IT Specialist": ["IT Support", "Hardware Maintenance", "Software Installation", "Network Administration", "Troubleshooting"],
    "Logistics Manager": ["Logistics Management", "Supply Chain Management", "Inventory Management", "Transportation Management", "Warehousing"],
    "Legal Advisor": ["Legal Advice", "Legal Research", "Document Drafting", "Client Consultation", "Litigation Support"],
    "Marketing Coordinator": ["Marketing Support", "Event Coordination", "Content Creation", "Social Media Management", "Email Marketing"],
    "Customer Service Supervisor": ["Customer Service", "Team Management", "Performance Monitoring", "Training", "Quality Assurance"],
    "Financial Controller": ["Financial Reporting", "Financial Analysis", "Budgeting", "Forecasting", "Financial Strategy"],
    "Operations Supervisor": ["Operations Management", "Team Leadership", "Process Improvement", "Workflow Management", "Quality Control"],
    "Legal Executive": ["Legal Documentation", "Document Drafting", "Contract Review", "Legal Research", "Legal Compliance"],
    "Marketing Executive": ["Marketing", "Campaign Management", "Content Creation", "Digital Marketing", "Social Media Management"],
    "Customer Service Associate": ["Customer Support", "Issue Resolution", "Order Processing", "Complaint Handling", "Call Handling"],
    "Financial Accountant": ["Financial Accounting", "Financial Reporting", "Taxation", "Audit", "Financial Analysis"],
    "Operations Administrator": ["Operations Support", "Administrative Assistance", "Project Coordination", "Data Entry", "Documentation"],
    "Legal Manager": ["Legal Management", "Team Leadership", "Legal Strategy", "Compliance Management", "Litigation Management"],
    "Marketing Director": ["Marketing Strategy", "Campaign Management", "Brand Development", "Digital Marketing", "Market Research"],
    "Customer Service Assistant": ["Customer Service", "Administrative Support", "Data Entry", "Order Processing", "Call Handling"],
    "Financial Analyst": ["Financial Analysis", "Financial Modeling", "Budgeting", "Forecasting", "Data Analysis"],
    "Operations Analyst": ["Operations Analysis", "Process Improvement", "Data Analysis", "Performance Monitoring", "Report Generation"],
    "Legal Secretary": ["Legal Assistance", "Document Preparation", "Client Communication", "Administrative Support", "Legal Research"],
    "Marketing Assistant": ["Marketing Support", "Content Creation", "Social Media Management", "Email Marketing", "Event Coordination"],
    "Customer Service Manager": ["Customer Service", "Team Management", "Problem-Solving", "Customer Satisfaction", "Quality Assurance"],
    "Financial Analyst": ["Financial Analysis", "Financial Modeling", "Data Analysis", "Excel", "Budgeting"],
    "Operations Analyst": ["Operations Analysis", "Data Analysis", "Process Improvement", "Report Generation", "Performance Monitoring"],
    "Legal Secretary": ["Legal Assistance", "Document Preparation", "Client Communication", "Administrative Support", "Legal Research"],
    "IT Technician": ["Technical Support", "Hardware Installation", "Software Installation", "Troubleshooting", "Network Configuration"],
    "Market Research Analyst": ["Market Research", "Data Analysis", "Survey Design", "Statistical Analysis", "Report Writing"],
    "Accounting Clerk": ["Accounting", "Bookkeeping", "Invoicing", "Financial Records", "Microsoft Excel"],
    "IT Administrator": ["System Administration", "Network Administration", "Server Management", "Security Management", "User Support"],
    "Logistics Coordinator": ["Logistics Management", "Inventory Management", "Supply Chain Coordination", "Transportation Management", "Shipping"],
    "Legal Consultant": ["Legal Consulting", "Legal Research", "Contract Review", "Legal Compliance", "Legal Writing"],
    "Marketing Specialist": ["Marketing Strategy", "Campaign Management", "Content Creation", "Digital Marketing", "Email Marketing"],
    "Customer Service Coordinator": ["Customer Service", "Complaint Handling", "Order Processing", "Data Entry", "Call Handling"],
    "Financial Planner": ["Financial Planning", "Investment Management", "Retirement Planning", "Wealth Management", "Tax Planning"],
    "Operations Assistant": ["Operations Support", "Administrative Assistance", "Project Coordination", "Data Entry", "Documentation"],
    "IT Analyst": ["IT Analysis", "System Analysis", "Data Analysis", "Software Testing", "Troubleshooting"],
    "Legal Analyst": ["Legal Research", "Legal Documentation", "Case Analysis", "Regulatory Compliance", "Legal Writing"],
    "Marketing Analyst": ["Marketing Analysis", "Data Analysis", "Market Research", "Campaign Performance Analysis", "Statistical Analysis"],
    "Customer Support Representative": ["Customer Support", "Technical Support", "Troubleshooting", "Ticketing Systems", "Communication Skills"],
    "Accounting Manager": ["Accounting", "Financial Reporting", "Budgeting", "Financial Analysis", "Team Management"],
    "IT Specialist": ["IT Support", "Hardware Maintenance", "Software Installation", "Network Administration", "Troubleshooting"],
    "Logistics Manager": ["Logistics Management", "Supply Chain Management", "Inventory Management", "Transportation Management", "Warehousing"],
    "Legal Advisor": ["Legal Advice", "Legal Research", "Document Drafting", "Client Consultation", "Litigation Support"],
    "Marketing Coordinator": ["Marketing Support", "Event Coordination", "Content Creation", "Social Media Management", "Email Marketing"],
    "Customer Service Supervisor": ["Customer Service", "Team Management", "Performance Monitoring", "Training", "Quality Assurance"],
    "Financial Controller": ["Financial Reporting", "Financial Analysis", "Budgeting", "Forecasting", "Financial Strategy"],
    "Operations Supervisor": ["Operations Management", "Team Leadership", "Process Improvement", "Workflow Management", "Quality Control"],
    "Legal Executive": ["Legal Documentation", "Document Drafting", "Contract Review", "Legal Research", "Legal Compliance"],
    "Marketing Executive": ["Marketing", "Campaign Management", "Content Creation", "Digital Marketing", "Social Media Management"],
    "Customer Service Associate": ["Customer Support", "Issue Resolution", "Order Processing", "Complaint Handling", "Call Handling"],
    "Financial Accountant": ["Financial Accounting", "Financial Reporting", "Taxation", "Audit", "Financial Analysis"],
    "Operations Administrator": ["Operations Support", "Administrative Assistance", "Project Coordination", "Data Entry", "Documentation"],
    "Legal Manager": ["Legal Management", "Team Leadership", "Legal Strategy", "Compliance Management", "Litigation Management"],
    "Marketing Director": ["Marketing Strategy", "Campaign Management", "Brand Development", "Digital Marketing", "Market Research"],
    "Customer Service Assistant": ["Customer Service", "Administrative Support", "Data Entry", "Order Processing", "Call Handling"],
    "Financial Analyst": ["Financial Analysis", "Financial Modeling", "Budgeting", "Forecasting", "Data Analysis"],
    "Operations Analyst": ["Operations Analysis", "Process Improvement", "Data Analysis", "Performance Monitoring", "Report Generation"],
    "Legal Secretary": ["Legal Assistance", "Document Preparation", "Client Communication", "Administrative Support", "Legal Research"],
    "Marketing Assistant": ["Marketing Support", "Content Creation", "Social Media Management", "Email Marketing", "Event Coordination"],
    "Customer Service Manager": ["Customer Service", "Team Management", "Problem-Solving", "Customer Satisfaction", "Quality Assurance"],
    "Financial Analyst": ["Financial Analysis", "Financial Modeling", "Data Analysis", "Excel", "Budgeting"],
    "Operations Analyst": ["Operations Analysis", "Data Analysis", "Process Improvement", "Report Generation", "Performance Monitoring"],
    "Legal Secretary": ["Legal Assistance", "Document Preparation", "Client Communication", "Administrative Support", "Legal Research"],
    "IT Technician": ["Technical Support", "Hardware Installation", "Software Installation", "Troubleshooting", "Network Configuration"],
    "Market Research Analyst": ["Market Research", "Data Analysis", "Survey Design", "Statistical Analysis", "Report Writing"],
    "Accounting Clerk": ["Accounting", "Bookkeeping", "Invoicing", "Financial Records", "Microsoft Excel"],
    "IT Administrator": ["System Administration", "Network Administration", "Server Management", "Security Management", "User Support"],
    "Logistics Coordinator": ["Logistics Management", "Inventory Management", "Supply Chain Coordination", "Transportation Management", "Shipping"],
    "Legal Consultant": ["Legal Consulting", "Legal Research", "Contract Review", "Legal Compliance", "Legal Writing"],
    "Marketing Specialist": ["Marketing Strategy", "Campaign Management", "Content Creation", "Digital Marketing", "Email Marketing"],
    "Customer Service Coordinator": ["Customer Service", "Complaint Handling", "Order Processing", "Data Entry", "Call Handling"],
    "Financial Planner": ["Financial Planning", "Investment Management", "Retirement Planning", "Wealth Management", "Tax Planning"],
    "Operations Assistant": ["Operations Support", "Administrative Assistance", "Project Coordination", "Data Entry", "Documentation"],
    "IT Analyst": ["IT Analysis", "System Analysis", "Data Analysis", "Software Testing", "Troubleshooting"],
    "Legal Analyst": ["Legal Research", "Legal Documentation", "Case Analysis", "Regulatory Compliance", "Legal Writing"],
    "Marketing Analyst": ["Marketing Analysis", "Data Analysis", "Market Research", "Campaign Performance Analysis", "Statistical Analysis"],
    "Customer Support Representative": ["Customer Support", "Technical Support", "Troubleshooting", "Ticketing Systems", "Communication Skills"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "React", "Node.js"],
    "Full Stack Developer": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Python", "Django", "MongoDB", "SQL"],
        "Python Developer": ["Python", "Flask", "Django", "FastAPI", "SQLAlchemy", "RESTful API Development", "Web Development", "Object-Oriented Programming (OOP)", "Database Management", "SQL", "ORM (Object-Relational Mapping)", "Version Control (e.g., Git)", "Unit Testing", "Integration Testing", "Debugging", "Problem-Solving"], 
        

    "ETL Developer": ["SQL", "ETL Tools (e.g., Informatica, Talend)", "Data Warehousing", "Data Modeling", "Scripting (e.g., Python)"],
    "Solution Architect": ["Architecture Design", "System Integration", "Cloud Services (e.g., AWS, Azure)", "Technical Leadership", "Project Management"],
    "Software Tester": ["Software Testing", "Test Automation", "Quality Assurance", "Bug Tracking", "Test Planning"],
    "Quality Assurance Analyst": ["Quality Assurance", "Test Case Design", "Manual Testing", "Defect Tracking", "Regression Testing"],
    "Automation Tester": ["Test Automation", "Selenium", "Python", "Java", "Test Frameworks (e.g., JUnit, TestNG)"],
    "Performance Tester": ["Performance Testing", "Load Testing", "Stress Testing", "Performance Monitoring", "Benchmarking"],
    "Database Tester": ["Database Testing", "SQL", "Data Integrity", "Data Migration", "Data Validation"],
    "Security Tester": ["Security Testing", "Vulnerability Assessment", "Penetration Testing", "Security Tools (e.g., Burp Suite, Metasploit)"],
    "ETL Tester": ["ETL Testing", "Data Validation", "Data Transformation", "ETL Tools (e.g., Informatica, Talend)", "SQL"],
    "System Tester": ["System Testing", "Integration Testing", "End-to-End Testing", "System Requirements Analysis", "Defect Tracking"],
    "Network Tester": ["Network Testing", "Protocol Analysis", "Network Troubleshooting", "Traffic Analysis", "Performance Testing"],
    "Regression Tester": ["Regression Testing", "Test Automation", "Test Scripting", "Continuous Integration", "Defect Tracking"],
    "Web Services Tester": ["Web Services Testing", "API Testing", "SOAP", "RESTful APIs", "Postman"],
    "UI/UX Tester": ["UI Testing", "UX Testing", "Usability Testing", "User Experience Analysis", "Cross-Browser Testing"],
    "Mobile App Tester": ["Mobile App Testing", "iOS", "Android", "Mobile Automation Tools (e.g., Appium)", "Performance Testing"],
    "Load Tester": ["Load Testing", "Performance Testing", "Stress Testing", "Scalability Testing", "Capacity Planning"],
        "Data Analyst": ["Data Analysis", "SQL", "Excel", "Data Visualization", "Statistical Analysis"]
}


# Function to clean and beautify JSON data
def clean_and_beautify_json(json_data):
    # Decode JSON string twice
    json_data = json.loads(json.loads(json_data))
    
    # Clean and validate each field
    for key, value in json_data.items():
        if isinstance(value, str):
            json_data[key] = value.strip()
        elif isinstance(value, list):
            json_data[key] = [item.strip() if isinstance(item, str) else item for item in value]
        elif isinstance(value, dict):
            json_data[key] = {sub_key: sub_value.strip() if isinstance(sub_value, str) else sub_value for sub_key, sub_value in value.items()}
    
    # Convert previousCompanyInformation from string to list of dictionaries
    if 'previousCompanyInformation' in json_data and isinstance(json_data['previousCompanyInformation'], str):
        json_data['previousCompanyInformation'] = json.loads(json_data['previousCompanyInformation'].replace("'", '"'))
    
    # Beautify the JSON data
    beautified_json_data = json.dumps(json_data, indent=4)
    return json_data, beautified_json_data

# Text Preprocessing
def preprocess_text(text):
    # Remove unnecessary characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Determine the best job role
def find_best_role(candidate_data):
    # Extract candidate skills
    candidate_skills = candidate_data.get('skills', "")
    
    # Extract job role and responsibilities from previous company information
    job_responsibilities = ""
    if 'previousCompanyInformation' in candidate_data:
        for company_info in candidate_data['previousCompanyInformation']:
            if 'jobRoleAndResponsibilities' in company_info and company_info['jobRoleAndResponsibilities']:
                job_responsibilities += company_info['jobRoleAndResponsibilities'] + " "
    
    # Preprocess candidate skills and job responsibilities
    candidate_skills_preprocessed = preprocess_text(candidate_skills)
    job_responsibilities_preprocessed = preprocess_text(job_responsibilities)
    
    # Combine job role skills into a single string
    job_roles_combined = {role: preprocess_text(' '.join(skills)) for role, skills in job_roles.items()}
    
    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    
    # Fit the vectorizer on the candidate skills and job role skills
    tfidf_matrix = tfidf_vectorizer.fit_transform([candidate_skills_preprocessed] + [job_responsibilities_preprocessed] + list(job_roles_combined.values()))
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[0:2], tfidf_matrix[2:]).flatten()
    
    # Check if there are valid similarities
    if len(cosine_similarities) == 0:
        return "No suitable job role found"
    
    # Find the index of the highest similarity score
    highest_score_index = cosine_similarities.argmax()
    
    # Find the corresponding job role
    best_role = list(job_roles.keys())[highest_score_index]
    
    return best_role

# Process the candidate data and find the best job role
def process_candidate_data(json_data):
    data, beautified_json_data = clean_and_beautify_json(json_data)
    
    # Find the best job role
    best_role = find_best_role(data)

    print("Beautified JSON Data:")
    print(beautified_json_data)
    print("\nBest Job Role for the Candidate:")
    print(f"Role: {best_role}")

# JSON Data
json_data = '''
"{\\"candidateName\\": \\"DOODA RAJESH\\", \\"address\\": \\"Visakhapatnam/Andhra Pradesh\\", \\"linkedUrl\\": \\"/Doodarajesh\\", \\"email\\": \\"rajesh251139@gmail.com\\", \\"phoneNumber\\": \\"8328570962\\", \\"higherQualification\\": \\"B.TECH COMPUTER SCIENCE - 71%\\", \\"passingYear\\": 2020, \\"skills\\": \\"\\\\\\"Python,Django,MongoDB,HTML,CSS,JavaScript,MySQL,PHP,WordPress,Git,GitHub,Bootstrap,Shellpro\\\\\\"\\", \\"city\\": \\"VISAKHAPATNAM\\", \\"certification\\": \\"\\\\u201cYellow chatbot developer\\\\u201d\\\\nYellow.ai provided by Yellow\\\\n\\\\u201cYellow Platform Developer Advance\\\\u201d\\\\nYellow.ai provided by Yellow\\", \\"previousCompanyInformation\\": \\"[{\\\\\\"Company_Name\\\\\\": \\\\\\"Vivify Healthcare Private Limited\\\\\\", \\\\\\"start_date\\\\\\": \\\\\\"Apr2021\\\\\\", \\\\\\"end_date\\\\\\": null, \\\\\\"Duration\\\\\\": null, \\\\\\"jobRoleAndResponsibilities\\\\\\": \\\\\\"\\\\\\\\u2022\\\\\\\\nMaintaining and updating thesites with team.\\\\\\\\n\\\\\\\\u2022Designed the web API using Django REST.\\\\\\\\n\\\\\\\\u2022Tech Used-Python,Django,Get , Post methods, etc.\\\\\\", \\\\\\"Project_Name\\\\\\": null}, {\\\\\\"Company_Name\\\\\\": \\\\\\"Bot\\\\\\", \\\\\\"start_date\\\\\\": \\\\\\"Jul 2018\\\\\\", \\\\\\"end_date\\\\\\": \\\\\\"Mar 2020\\\\\\", \\\\\\"Duration\\\\\\": 22, \\\\\\"jobRoleAndResponsibilities\\\\\\": \\\\\\"\\\\\\\\u2022Crud operation and database connection.\\\\\\", \\\\\\"Project_Name\\\\\\": null}, {\\\\\\"Company_Name\\\\\\": \\\\\\"Haritha Computer private limited\\\\\\", \\\\\\"start_date\\\\\\": \\\\\\"Jan 2016\\\\\\", \\\\\\"end_date\\\\\\": \\\\\\"Sept 2016\\\\\\", \\\\\\"Duration\\\\\\": 9, \\\\\\"jobRoleAndResponsibilities\\\\\\": \\\\\\"\\\\\\\\u2022System repair.\\\\\\", \\\\\\"Project_Name\\\\\\": null}]\\", \\"totalNumberOfExperience\\": 31, \\"lastCompany\\": \\"Bot\\", \\"maximumWorkDuration\\": 22, \\"minimumWorkDuration\\": 9, \\"averageDuration\\": 17}"
'''

# Process candidate data and find the best job role
process_candidate_data(json_data)
