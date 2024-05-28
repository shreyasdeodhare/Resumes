import json
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
tfidf_vectorizer = TfidfVectorizer()

# Define the job roles and skills
job_roles = {
    "Data Scientist": ["Python", "R", "SQL", "Machine Learning", "Data Analysis"],
    "Software Engineer": ["Java", "JavaScript", "Python", "SQL", "HTML/CSS"],
    "Full Stack Developer": ["JavaScript", "Node.js", "React", "Angular", "MongoDB"],
    # Add more job roles and skills here...
}

# Function to clean and preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove unnecessary characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Function to prepare data for model training
def prepare_data():
    X = []
    y = []
    for role, skills in job_roles.items():
        X.extend([preprocess_text(' '.join(skills))] * 100)  # Replicate examples for each role
        y.extend([role] * 100)
    return X, y

# Train the model
def train_model(X_train, y_train):
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)
    return model

# Function to predict job role
def predict_job_role(model, tfidf_vectorizer,candidate_data):
    candidate_text = preprocess_text(candidate_data)
    candidate_tfidf = tfidf_vectorizer.transform([candidate_text])
    predicted_role = model.predict(candidate_tfidf)[0]
    return predicted_role

# Example candidate data
candidate_data = '''
{
  "jobRoleAndResponsibilities": "Maintain and update websites with team. Design web APIs using Django REST.",
  "skills": "Python, Django, HTML, CSS, JavaScript, MySQL, PHP"
}
'''

# Prepare training data
X, y = prepare_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = train_model(X_train, y_train)

# Evaluate model accuracy
X_val_tfidf = tfidf_vectorizer.transform(X_val)
y_pred = model.predict(X_val_tfidf)
accuracy = accuracy_score(y_val, y_pred)
print("Model Accuracy:", accuracy)

# Predict job role for candidate data
predicted_role = predict_job_role(model, tfidf_vectorizer,candidate_data)
print("Predicted Job Role:", predicted_role)
