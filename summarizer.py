import streamlit as st
import spacy
import pandas as pd
import docx2txt
import re
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Load the SpaCy NER model
nlp_ner = spacy.load("./model")

# Load NLTK resources
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

# Streamlit App
st.title("Resume Parser and Summarizer")

# File upload widget for multiple files
uploaded_files = st.file_uploader("Upload Multiple Files (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            # Read the uploaded file
            file_extension = uploaded_file.name.split('.')[-1]

            if file_extension == 'pdf':
                # Read the PDF file using PyMuPDF
                pdf_data = uploaded_file.read()
                pdf_document = PyMuPDF.PDFDocument(PyMuPDF.MemoryPDFDoc(pdf_data))
                text = ""
                for page in pdf_document:
                    text += page.getText()

            elif file_extension == 'docx':
                # Read the DOCX file
                text = docx2txt.process(uploaded_file)

            # Process the text with the SpaCy NER model
            parsed_doc = nlp_ner(text)

            # Extract the content from the resume
            extracted_content = "\n".join([ent.text for ent in parsed_doc.ents])

            # Display the extracted content
            st.header("Extracted Content from Resume:")
            st.write(extracted_content)

            # Summarize the extracted content
            parser = PlaintextParser.from_string(extracted_content, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, sentences_count=3)

            # Display the summary
            st.header("Summary of Extracted Content:")
            for sentence in summary:
                st.write(sentence)

        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
