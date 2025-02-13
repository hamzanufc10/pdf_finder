import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Set up Google API key
api_key = "HAMZA"  # Replace with your actual API key
genai.configure(api_key=api_key)

# Function to scrape text from a URL
def get_webpage_text(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return f"Error: Unable to fetch page (Status Code {response.status_code})"
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract only the main content
        content_div = soup.find("div", {"id": "bodyContent"})  # Wikipedia-specific
        text = content_div.get_text(separator="\n", strip=True) if content_div else soup.get_text(separator="\n", strip=True)
        return text
    except Exception as e:
        return f"Error: {str(e)}"

# Split text into chunks for embeddings
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

# Store text chunks in FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Conversational chain setup
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say "The answer is not available in the context."
    
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Search FAISS and respond to user query
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Reply: ", response["output_text"])

# Streamlit UI
def main():
    st.set_page_config("Chat with Webpages")
    st.header("Chat with Webpage Content using Gemini AI 🌐")

    url = st.text_input("Enter the URL of the webpage:")
    
    if st.button("Fetch & Process"):
        with st.spinner("Fetching webpage content..."):
            raw_text = get_webpage_text(url)
            if "Error" in raw_text:
                st.error(raw_text)
            else:
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Webpage content processed successfully!")

    user_question = st.text_input("Ask a question about the webpage content:")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
