from transformers import BartTokenizer, BartForConditionalGeneration
import streamlit as st
import fitz
import io

# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Function to extract text from uploaded PDF file
def extract_pdf_text(uploaded_file):
    try:
        # Convert the uploaded PDF file to bytes
        pdf_bytes = io.BytesIO(uploaded_file.read())

        # Create a document object
        doc = fitz.open(stream=pdf_bytes)

        # Extract text from each page
        text = ""
        for page in doc:
            text += page.get_text() + "\n"

        return text

    except Exception as e:
        st.error(f"An error occurred while extracting text: {str(e)}")
        return None

# Streamlit App Title
st.title("Resume Text Summarizer")

# File Upload Widget
uploaded_file = st.file_uploader("Upload PDF File", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF file
    pdf_text = extract_pdf_text(uploaded_file)

    if pdf_text:
        # Tokenize and summarize the input text using BART
        inputs = tokenizer.encode("summarize: " + pdf_text, return_tensors="pt", max_length=4096, truncation=False)
        summary_ids = model.generate(inputs, max_length=500, min_length=100, length_penalty=1.0, num_beams=3, early_stopping=False)

        # Decode and output the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Display the original text and the summary
        st.header("Original Text:")
        st.text_area("PDF Text", pdf_text, height=250)

        st.header("Summary:")
        st.text_area("Summary", summary, height=250)
