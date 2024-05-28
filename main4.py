import json
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
    return tokens

# Normalize responsibilities and skills
normalized_responsibilities = [preprocess_text(responsibility) for responsibility in responsibilities]
normalized_skills = preprocess_text(skills)

# Flatten the list of responsibilities
all_responsibilities = [word for sublist in normalized_responsibilities for word in sublist]

# Semantic Analysis using TF-IDF and Cosine Similarity
def calculate_similarity(responsibilities, skills):
    # Concatenate all responsibilities into a single string
    responsibilities_text = [' '.join(responsibility) for responsibility in responsibilities]
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    # Fit and transform TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(responsibilities_text + [' '.join(skills)])
    # Calculate cosine similarity between the TF-IDF vectors
    similarity_matrix = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1:])
    # Get top semantic matches
    top_matches_indices = similarity_matrix.argsort(axis=None)[::-1][:len(skills)]
    top_matches = [responsibilities_text[i] for i in top_matches_indices]
    return top_matches

# Calculate semantic similarity
semantic_matches = calculate_similarity(normalized_responsibilities, normalized_skills)

# Display Results
print("Top semantic matches for skills:")
for match in semantic_matches:
    print(match)
