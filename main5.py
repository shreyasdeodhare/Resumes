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
    data = json.loads(json_data)
    
    # Clean and validate each field
    for key, value in data.items():
        if isinstance(value, str):
            data[key] = value.strip()
        elif isinstance(value, list):
            data[key] = [item.strip() if isinstance(item, str) else item for item in value]
        elif isinstance(value, dict):
            data[key] = {sub_key: sub_value.strip() if isinstance(sub_value, str) else sub_value for sub_key, sub_value in value.items()}

    # Beautify the JSON data
    beautified_json_data = json.dumps(data, indent=4)
    return data, beautified_json_data

# Text Preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Determine the best job role
def find_best_role(candidate_data):
    # Check if job role is directly mentioned in the JSON
    if 'jobRoleAndResponsibilities' in candidate_data and candidate_data['jobRoleAndResponsibilities']:
        candidate_role = candidate_data['jobRoleAndResponsibilities']
        for role in job_roles.keys():
            if role.lower() in candidate_role.lower():
                return role
    
    # Extract candidate skills
    candidate_skills = candidate_data['skills']
    
    # Preprocess candidate skills
    candidate_skills_preprocessed = preprocess_text(candidate_skills)
    
    # Combine job role skills into a single string
    job_roles_combined = {role: preprocess_text(' '.join(skills)) for role, skills in job_roles.items()}
    
    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    
    # Fit the vectorizer on the candidate skills and job role skills
    tfidf_matrix = tfidf_vectorizer.fit_transform([candidate_skills_preprocessed] + list(job_roles_combined.values()))
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
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

# Example candidate data for each role (You can add more examples for additional roles)
candidate_data_examples = {
    "Data Scientist": '''
{
  "skills": "Python, R, SQL, Machine Learning, Data Analysis"
}
''',
    "Software Engineer": '''
{
  "skills": "Java, JavaScript, Python, SQL, HTML/CSS"
}
''',
    "Full Stack Developer": '''
{
  "skills": "JavaScript, Node.js, React, Angular, MongoDB"
}
''',
    "Product Manager": '''
{
  "skills": "Product Management, Agile Methodologies, UX/UI Design, Market Research, Project Management"
}
''',
    "Business Analyst": '''
{
  "skills": "Business Analysis, Requirements Gathering, Data Analysis, SQL, Project Management"
}
''',
    "Digital Marketing Specialist": '''
{
  "skills": "SEO, SEM, Social Media Marketing, Google Analytics, Content Marketing"
}
''',
    "Sales Executive": '''
{
  "skills": "Sales, Customer Relationship Management (CRM), Negotiation, Communication Skills, Lead Generation"
}
''',
    "Marketing Manager": '''
{
  "skills": "Marketing Strategy, Brand Management, Market Research, Digital Marketing, Content Creation"
}
''',
    "Quality Assurance Engineer": '''
{
  "skills": "Manual Testing, Automated Testing, Test Planning, Bug Tracking, Agile Methodologies"
}
''',
    "Network Engineer": '''
{
  "skills": "Networking, Cisco, LAN/WAN, Routing and Switching, Network Security"
}
''',
    "UI/UX Designer": '''
{
  "skills": "User Interface Design, User Experience Design, Wireframing, Prototyping, Adobe Creative Suite"
}
''',
    "DevOps Engineer": '''
{
  "skills": "Linux, Docker, Kubernetes, CI/CD, Infrastructure as Code"
}
''',
    "Cybersecurity Analyst": '''
{
  "skills": "Cybersecurity, Penetration Testing, Firewall Management, Incident Response, Security Operations"
}
''',
    "Financial Analyst": '''
{
  "skills": "Financial Modeling, Financial Reporting, Data Analysis, Excel, Financial Planning"
}
''',
    "Content Writer": '''
{
  "skills": "Content Writing, Copywriting, SEO Writing, Content Strategy, Editing"
}
''',
    "Human Resources Manager": '''
{
  "skills": "Recruitment, Employee Relations, Performance Management, HR Policies, Training and Development"
}
''',
    "Graphic Designer": '''
{
  "skills": "Adobe Photoshop, Illustrator, InDesign, Typography, Visual Communication"
}
''',
    "Project Manager": '''
{
  "skills": "Project Management, Risk Management, Stakeholder Management, Agile Methodologies, Budgeting"
}
''',
    "Operations Manager": '''
{
  "skills": "Operations Management, Process Improvement, Supply Chain Management, Logistics, Inventory Management"
}
''',
    "Software Architect": '''
{
  "skills": "Software Architecture, System Design, Microservices, Cloud Computing, Design Patterns"
}
''',
    "Data Engineer": '''
{
  "skills": "Python, SQL, Big Data, Hadoop, Spark, Kafka, ETL, Data Warehousing, Airflow, NoSQL, AWS, Azure, Google Cloud Platform"
}
''',
    "Technical Support Engineer": '''
{
  "skills": "Troubleshooting, Customer Support, Networking, Operating Systems, Ticketing Systems"
}
''',
    "UX Researcher": '''
{
  "skills": "User Research, Usability Testing, Qualitative Research, Quantitative Research, Psychology"
}
''',
    "Mobile App Developer": '''
{
  "skills": "Android, iOS, Swift, Java, Mobile UI Design"
}
''',
    "Cloud Solutions Architect": '''
{
  "skills": "AWS, Azure, Google Cloud Platform, Serverless Architecture, DevOps"
}
''',
    "Supply Chain Manager": '''
{
  "skills": "Supply Chain Optimization, Inventory Management, Procurement, Logistics, Supplier Relationship Management"
}
'''
}

# Process candidate data and find the best job role for each example
for role, data_json in candidate_data_examples.items():
    print(f"\nTesting {role} Classification:")
    cleaned_json_data = re.sub(r'\\', '', data_json)
    process_candidate_data(cleaned_json_data)

