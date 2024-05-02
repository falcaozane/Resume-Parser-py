import streamlit as st
import PyPDF2
from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline('summarization')

st.title('PDF Summarizer')

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # Open the uploaded PDF file
    with open(uploaded_file, 'rb') as file:
        # Create a PDF reader
        reader = PyPDF2.PdfFileReader(file)

        # Initialize an empty string to hold the text
        text = ""

        # Loop through each page in the PDF and extract the text
        for page in range(reader.numPages):
            text += reader.getPage(page).extractText()

    # Summarize the extracted text
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)

    # Display the summary
    st.header('Summary:')
    st.write(summary[0]['summary_text'])
