import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load JSON data
json_data = '''
{
    "gender": "M",
    "address": "Solanke Layout, Ward No. 12, Buldana, Tq Buldana, Di Buldana 443001.",
    "email": "aakash.s.bhalerao358@gmail.com",
    "skills": "HTML5, CSS, SQL, C, C++, Bootstrap, JavaScript Animation Libraries, MySQL, Microsoft Office suite, Adobe Photoshop (Beginner), Windows (7, 8, 10) user, Android Studio, Visual Studio, VS Code, Sublime Text. Linux (user level).",
    "city": "BULDANA",
    "candidateName": "Aakash Bhalerao",
    "dateOfBirth": "24-September-1992",
    "phoneNumber": "+91-8552939143",
    "altPhoneNumber": "+91-8623074807",
    "higherQualification": "BACHELOR OF ENGINEERING",
    "passingYear": "2020",
    "previousCompanyInformation": [
        {
            "Company_Name": "Prostorm Innotech Pvt. Ltd., Buldana",
            "start_date": "April 2020",
            "end_date": null,
            "Duration": "28",
            "jobRoleAndResponsibilities": "Requirements Gathering for Website/Web Application Development. Analysis, Planning, Designing, Developing Websites and Web Applications. Hosting and Maintenance of Websites and Web Applications.",
            "Project_Name": null
        },
        {
            "Company_Name": "Prostorm Innotech Pvt. Ltd., Buldana",
            "start_date": "October-2019",
            "end_date": "March-2020",
            "Duration": "6",
            "jobRoleAndResponsibilities": null,
            "Project_Name": null
        }
    ],
    "currentCompanyName": "Prostorm Innotech Pvt. Ltd., Buldana",
    "Classification": "Web Developer",
    "lastCompany": "Prostorm Innotech Pvt. Ltd., Buldana",
    "maximumWorkDuration": "28",
    "minimumWorkDuration": "6"
}
'''

# Parse JSON
data = json.loads(json_data)

# Extract job responsibilities
responsibilities = []
for company_info in data["previousCompanyInformation"]:
    if "jobRoleAndResponsibilities" in company_info and company_info["jobRoleAndResponsibilities"]:
        responsibilities.append(company_info["jobRoleAndResponsibilities"])

# Extract skills
skills = data["skills"]

# Tokenization
responsibilities_tokens = word_tokenize(' '.join(responsibilities))
skills_tokens = word_tokenize(skills)

# Remove stopwords
stop_words = set(stopwords.words('english'))
responsibilities_filtered = [word for word in responsibilities_tokens if word.lower() not in stop_words]
skills_filtered = [word for word in skills_tokens if word.lower() not in stop_words]

# Lemmatization
lemmatizer = WordNetLemmatizer()
responsibilities_lemmatized = [lemmatizer.lemmatize(word) for word in responsibilities_filtered]
skills_lemmatized = [lemmatizer.lemmatize(word) for word in skills_filtered]

# Convert to comma-separated strings
responsibilities_str = ' '.join(responsibilities_lemmatized)
skills_str = ' '.join(skills_lemmatized)

print("Responsibilities:", responsibilities_str)
print("Skills:", skills_str)
