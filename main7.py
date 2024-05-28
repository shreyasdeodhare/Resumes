import json
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define the job roles and skills
job_roles = {
    "Data Scientist": ["Python", "R", "SQL", "Machine Learning", "Data Analysis"],
    "Software Engineer": ["Java", "JavaScript", "Python", "SQL", "HTML/CSS"],
    "Full Stack Developer": ["JavaScript", "Node.js", "React", "Angular", "MongoDB"],
    "Python backend Developer":["Python","Django REST","API","MySQL"]
    
    # Add more job roles and skills here...
}

# Function to clean and preprocess text
def preprocess_text(text):
    """
    Preprocesses the input text by converting to lowercase, removing unnecessary characters,
    tokenizing, removing stopwords, and lemmatizing.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove unnecessary characters
    tokens = word_tokenize(text)  # Tokenization
    stop_words = set(stopwords.words('english'))  # Load stopwords
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

# Function to prepare data for model training
def prepare_data(job_roles):
    """
    Prepares the training data by replicating examples for each job role and preprocessing the skills.
    """
    X = []
    y = []
    for role, skills in job_roles.items():
        processed_text = preprocess_text(' '.join(skills))
        X.extend([processed_text] * 100)  # Replicate examples for each role
        y.extend([role] * 100)
    return X, y

# Train the model
def train_model(X_train, y_train):
    """
    Trains a RandomForest model using TF-IDF features extracted from the training data.
    """
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)
    return model, tfidf_vectorizer

# Function to predict job role
def predict_job_role(model, tfidf_vectorizer, candidate_data):
    """
    Predicts the job role for the given candidate data using the trained model and TF-IDF vectorizer.
    """
    candidate_dict = json.loads(candidate_data)
    candidate_text = preprocess_text(candidate_dict.get('jobRoleAndResponsibilities', '') + ' ' + candidate_dict.get('skills', ''))
    candidate_tfidf = tfidf_vectorizer.transform([candidate_text])
    predicted_role = model.predict(candidate_tfidf)[0]
    return predicted_role

# Main function to execute the script
def main():
    # Example candidate data
    candidate_data = '''
    {
      "jobRoleAndResponsibilities": "Maintain and update websites with team. Design web APIs using Django REST.",
      "skills": "Python, Django, HTML, CSS, JavaScript, MySQL, PHP"
    }
    '''

    try:
        # Prepare training data
        X, y = prepare_data(job_roles)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model and get the TF-IDF vectorizer
        model, tfidf_vectorizer = train_model(X_train, y_train)

        # Evaluate model accuracy
        X_val_tfidf = tfidf_vectorizer.transform(X_val)
        y_pred = model.predict(X_val_tfidf)
        accuracy = accuracy_score(y_val, y_pred)
        print("Model Accuracy:", accuracy)

        # Predict job role for candidate data
        predicted_role = predict_job_role(model, tfidf_vectorizer, candidate_data)
        print("Predicted Job Role:", predicted_role)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
