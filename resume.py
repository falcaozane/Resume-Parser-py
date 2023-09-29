import streamlit as st
import re
import nltk
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the resume dataset
resumeDataSet = pd.read_csv('UpdatedResumeDataSet.csv', encoding='utf-8')

# Load NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Clean resume text
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)
    resumeText = re.sub('RT|cc', ' ', resumeText)
    resumeText = re.sub('#\S+', '', resumeText)
    resumeText = re.sub('@\S+', '  ', resumeText)
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)
    return resumeText

resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))

# Label encoding for 'Category'
var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resumeDataSet[i] = le.fit_transform(resumeDataSet[i])

# Tfidf Vectorization
requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)

# K-Nearest Neighbors Classifier
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)

# Streamlit App
st.title("Resume Category Classifier")

# File upload widget
uploaded_file = st.file_uploader("Upload a PDF or Text file", type=["pdf", "txt"])

if uploaded_file is not None:
    try:
        resume_bytes = uploaded_file.read()
        resume_text = resume_bytes.decode('utf-8')
    except UnicodeDecodeError:
        resume_text = resume_bytes.decode('latin-1')

    cleaned_resume = cleanResume(resume_text)
    input_features = word_vectorizer.transform([cleaned_resume])
    prediction_id = clf.predict(input_features)[0]
    
    category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
    }
    
    category_name = category_mapping.get(prediction_id, "Unknown")
    
    st.write("Predicted Category:", category_name)
