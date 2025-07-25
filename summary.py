import streamlit as st
import fitz
import spacy
import pandas as pd
import docx2txt
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration

# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Load the SpaCy NER model
nlp_ner = spacy.load("./model")

# Load NLTK resources
nltk.download('punkt')

# Function to clean resume text
def clean_resume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)
    resumeText = re.sub('RT|cc', ' ', resumeText)
    resumeText = re.sub('#\S+', '', resumeText)
    resumeText = re.sub('@\S+', '  ', resumeText)
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)
    return resumeText

# Load resume dataset
resumeDataSet = pd.read_csv('UpdatedResumeDataSet.csv', encoding='utf-8')

# Clean the resume text
resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: clean_resume(x))

# Label encoding for 'Category'
le = LabelEncoder()
resumeDataSet['Category'] = le.fit_transform(resumeDataSet['Category'])

# Tfidf Vectorization
requiredText = resumeDataSet['cleaned_resume'].values
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
WordFeatures = word_vectorizer.fit_transform(requiredText)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(WordFeatures, resumeDataSet['Category'], random_state=0, test_size=0.85)

# K-Nearest Neighbors Classifier
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)

# Function to parse and classify resume
def parse_and_classify_resume():
    st.title("Resume Parser and Category Classifier")

    # File upload widget for multiple files
    uploaded_files = st.file_uploader("Upload Multiple Files (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Read the uploaded file
                file_extension = uploaded_file.name.split('.')[-1]

                if file_extension == 'pdf':
                    # Read the PDF file
                    pdf_data = uploaded_file.read()
                    doc = fitz.open("uploaded.pdf", pdf_data)
                    text = " ".join([page.get_text() for page in doc])

                elif file_extension == 'docx':
                    # Read the DOCX file
                    text = docx2txt.process(uploaded_file)

                # Process the text with the SpaCy NER model
                parsed_doc = nlp_ner(text)

                # Display the extracted entities
                st.header("Extracted Entities from Resume:")
                for ent in parsed_doc.ents:
                    st.write(f"{ent.label_} : {ent.text}")

                st.divider()

                # Create a list of entities and labels
                entities = [ent.text for ent in parsed_doc.ents]
                labels = [ent.label_ for ent in parsed_doc.ents]

                # Create a Pandas DataFrame
                data = {'Label': labels, 'Content': entities}
                df = pd.DataFrame(data)

                # Display the extracted entities in a table
                st.header("Extracted Entities in Tabular format:")
                st.dataframe(df)

                # Predict the category using the K-Nearest Neighbors Classifier
                cleaned_resume = clean_resume(text)
                input_features = word_vectorizer.transform([cleaned_resume])
                prediction_id = clf.predict(input_features)[0]

                category_mapping = {
                0: "Advocate",
                1: "Arts",
                2: "Automation Testing",
                3: "Blockchain",
                4: "Business Analyst",
                5: "Civil Engineer",
                6: "Data Science",
                7: "Database",
                8: "DevOps Engineer",
                9: ".Net Developer",
                10: "ETL Developer",
                11: "Electrical Engineering",
                12: "HR",
                13: "Hadoop and Big Data",
                14: "Health and Fitness",
                15: "Java Developer",
                16: "Mechanical Engineer",
                17: "Network Security Engineer",
                18: "Operations Manager",
                19: "PMO",
                20: "Python Developer",
                21: "SAP Developer",
                22: "Sales",
                23: "Testing",
                24: "Web Designing / Frontend Developer",
                25: "Unknown"  # Add a category for unrecognized resumes
        }

                category_name = category_mapping.get(prediction_id, "Unknown")

                st.write("Predicted Category for Resume:", category_name)

            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")

# Function to summarize resume
def summarize_resume():
    st.title("Resume Summarizer")

    # File upload widget for multiple files
    uploaded_files = st.file_uploader("Upload Multiple Files (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Read the uploaded file
                file_extension = uploaded_file.name.split('.')[-1]

                if file_extension == 'pdf':
                    # Read the PDF file
                    pdf_data = uploaded_file.read()
                    doc = fitz.open("uploaded.pdf", pdf_data)
                    text = " ".join([page.get_text() for page in doc])

                elif file_extension == 'docx':
                    # Read the DOCX file
                    text = docx2txt.process(uploaded_file)

                # Process the text with the SpaCy NER model
                parsed_doc = nlp_ner(text)

                # Create a list of extracted content
                extracted_content = [ent.text for ent in parsed_doc.ents]

                # Join the extracted content into a single string
                doc_content = ' '.join(extracted_content)

                # Clean the extracted content
                cleaned_content = clean_resume(doc_content)

                # Summarize the cleaned content using BART
                inputs = tokenizer.encode("summarize: " + cleaned_content, return_tensors="pt", max_length=4096, truncation=True)
                summary_ids = model.generate(inputs, max_length=500, min_length=100, length_penalty=1.0, num_beams=4, early_stopping=True)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                # Output the summary
                st.header("Summary of Extracted Content:")
                st.write(summary)

            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")


# Sidebar navigation
selected_option = st.sidebar.selectbox(
    "Select Option:",
    ("Resume Parser and Category Classifier", "Resume Summarizer")
)

# Call the appropriate function based on the selected option
if selected_option == "Resume Parser and Category Classifier":
    parse_and_classify_resume()
elif selected_option == "Resume Summarizer":
    summarize_resume()
