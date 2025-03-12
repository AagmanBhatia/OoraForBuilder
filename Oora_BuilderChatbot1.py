import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, util
#from huggingface_hub import snapshot_download
from groq import Groq
import os


# --- Configuration ---
# Replace these with the actual file path and API key provided by your side.
PDF_FILE_PATH = "Builder Faqs.pdf"
#GROQ_API_KEY = ""

# --- Initialize Streamlit page ---
st.set_page_config(page_title="Oora", layout="wide")


st.image("Group 1597882494 - Copy.png", use_container_width=True, width=200)
st.image("poweredbymrproptek_img.png", use_container_width=True, width=170)
#st.title("Oora")
st.title("Say hi! to Oora, AI Companion for your queries")


# --- Initialize Groq API client and embedding model ---
#client = Groq(api_key=GROQ_API_KEY)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
embedding_model = load_embedding_model()

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_path: str) -> str:
    """Extracts text from all pages of the PDF."""
    text = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 2000) -> list:
    """Splits the text into smaller chunks to handle large documents."""
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk += "\n" + para
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def answer_query(user_query: str, pdf_chunks: list) -> str:
    """Selects the most relevant text chunk and uses the Groq API to generate an answer."""
    query_embedding = embedding_model.encode(user_query, convert_to_tensor=True)
    chunk_embeddings = [embedding_model.encode(chunk, convert_to_tensor=True) for chunk in pdf_chunks]
    similarities = [util.pytorch_cos_sim(query_embedding, chunk_emb).item() for chunk_emb in chunk_embeddings]
    most_similar_index = similarities.index(max(similarities))
    relevant_chunk = pdf_chunks[most_similar_index]

    messages = [
        {"role": "system", "content": "You are a super friendly and a quirky realestate assistant for builders/developers/mrproptek that answers questions based on the provided PDF document."},
        {"role": "system", "content": relevant_chunk},
        {"role": "user", "content": user_query},
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

# --- Process the PDF ---
#st.header("Processing PDF")
with st.spinner("Just Give me a second to get ready! I am still learning..."):
    try:
        pdf_text = extract_text_from_pdf(PDF_FILE_PATH)
        pdf_chunks = chunk_text(pdf_text)
        st.success("Yup! I am ready")
    except Exception as e:
        st.error(f"Error processing! Kindly Wait or contact info@mrproptek.com: {e}")
        pdf_chunks = []

# --- Query Input ---
st.header("Ask Your Question")
user_query = st.text_input("Enter your question:")
if st.button("Get Answer") and user_query:
    if pdf_chunks:
        with st.spinner("Oora is Ooring..."):
            answer = answer_query(user_query, pdf_chunks)
        st.markdown("### Answer")
        st.write(answer)
    else:
        st.error("No PDF content available to generate an answer.")
