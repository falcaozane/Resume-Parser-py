import streamlit as st
import fitz
import spacy
import spacy_transformers
import pandas as pd
import docx2txt
import re
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

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

                category_mapping = dict(zip(range(len(le.classes_)), le.classes_))

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

                # Summarize the cleaned content
                parser = PlaintextParser.from_string(cleaned_content, Tokenizer("english"))
                summarizer = LsaSummarizer()
                summary = summarizer(parser.document, sentences_count=3)

                # Output the summary
                st.header("Summary of Extracted Content:")
                for sentence in summary:
                    st.write(sentence)

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
