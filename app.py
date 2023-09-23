import streamlit as st
import fitz
import spacy
import pandas as pd
import docx2txt
# import frontend

# Load the SpaCy NER model
nlp = spacy.load('./model')

# Streamlit app title and description
st.title("Resume Parser")
st.write("Upload a PDF, DOCX  file containing a resume for parsing.")

# File upload widget
uploaded_file = st.file_uploader("Upload a File", type=["pdf", "docx"])

if uploaded_file is not None:
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
    parsed_doc = nlp(text)
    
    # Display the extracted entities
    st.header("Extracted Entities:")
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
