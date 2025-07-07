import os
import requests
import openai
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys and IDs from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI API key for older LangChain components and direct OpenAI calls
openai.api_key = OPENAI_API_KEY

# Global storage path for the FAISS vector store index
INDEX_PATH = "rag_index"

# --- Google Search and GPT Summarization Functions ---

def google_search(query):
    """
    Performs a Google Custom Search for a given query.
    Returns a list of search results items or None if the API request fails.
    """
    params = {
        'q': query,
        'key': GOOGLE_API_KEY,
        'cx': SEARCH_ENGINE_ID
    }
    try:
        response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        items = data.get('items', [])
        print(f"Retrieved {len(items)} search results from Google.")
        return items
    except requests.exceptions.RequestException as e:
        print(f"❌ Google Search API failed: {e}")
        return None

def summarize_with_gpt(snippets, user_query):
    """
    Uses OpenAI's GPT-4 to summarize provided text snippets in response to a user query.
    """
    combined_snippets = "\n".join(snippets)
    prompt = f"Using the following search results, answer the user's question:\n\n{combined_snippets}\n\nUser Question: {user_query}\nAnswer:"

    try:
        # Use the newer OpenAI client syntax (v1.x)
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except openai.APIError as e:
        print(f"❌ OpenAI API failed during summarization: {e}")
        return "An error occurred while summarizing with GPT."

# --- Selenium Web Scraper ---

def selenium_scrape(url):
    """
    Scrapes the text content of a given URL using Selenium in headless Chrome.
    Handles JavaScript rendering by waiting for a few seconds.
    """
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")        # Run Chrome in headless mode (no UI)
        chrome_options.add_argument("--disable-gpu")     # Required for some Linux environments
        chrome_options.add_argument("--no-sandbox")      # Required for some containerized environments
        chrome_options.add_argument("--disable-dev-shm-usage") # Overcomes limited resource problems

        # Initialize WebDriver
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(5)  # Allow time for JavaScript to render content
        text = driver.find_element(By.TAG_NAME, "body").text
        driver.quit() # Close the browser
        return text
    except Exception as e:
        print(f"❌ Selenium scraping failed for {url}: {e}")
        return ""

# --- RAG Data Collection and Management ---

def scrape_and_chunk(urls):
    """
    Scrapes content from a list of URLs, chunks the text, creates embeddings,
    and saves them to a FAISS vector store.
    """
    all_chunks = []
    for url in urls:
        print(f"🔗 Scraping: {url}")
        text = selenium_scrape(url)
        # Skip URLs with very little text, as they might be empty or error pages
        if len(text) < 300:
            print(f"⚠️ Skipped {url}: too little text ({len(text)} characters).")
            continue
        print(f"✅ Scraped {len(text)} characters from: {url}")

        # Initialize text splitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        # Create documents from the scraped text
        docs = splitter.create_documents([text])
        # Add source metadata to each document chunk
        for doc in docs:
            doc.metadata = {"source": url}
        all_chunks.extend(docs)

    if not all_chunks:
        st.warning("No valid content was scraped to create the RAG index.")
        return

    # Create embeddings using OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # Create FAISS vector store from documents and embeddings
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    # Save the vector store locally for later use
    vectorstore.save_local(INDEX_PATH)
    print(f"✅ RAG index created and saved to {INDEX_PATH}.")

def load_vectorstore():
    """
    Loads the FAISS vector store from the local path if it exists.
    Returns the loaded vector store or None if the index file is not found.
    """
    if not os.path.exists(INDEX_PATH):
        print(f"⚠️ RAG index not found at {INDEX_PATH}.")
        return None
    try:
        # Load the vector store, allowing dangerous deserialization as it's from a trusted source (our own creation)
        vectorstore = FAISS.load_local(
            INDEX_PATH,
            OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
            allow_dangerous_deserialization=True
        )
        print(f"✅ RAG index loaded from {INDEX_PATH}.")
        return vectorstore
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        return None

def ask_rag(question):
    """
    Queries the RAG vector store for an answer to the given question.
    Returns the answer and a list of sources.
    """
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return None, None # Cannot answer from RAG if vector store is not loaded

    # Initialize LangChain's OpenAI LLM
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0) # temperature=0 for more deterministic answers
    # Create a RetrievalQA chain for question answering
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 similar documents
    )
    try:
        response = qa_chain({"query": question})
        answer = response['result']
        # Get sources from the documents retrieved by the retriever
        # For simplicity, we get sources from a fresh similarity search.
        retrieved_docs = vectorstore.similarity_search(question, k=3)
        sources = list(set([doc.metadata['source'] for doc in retrieved_docs if 'source' in doc.metadata]))
        return answer, sources
    except Exception as e:
        print(f"❌ Error during RAG query: {e}")
        return "An error occurred while querying RAG.", None

# --- LLM-based Relevance Check Function ---

def is_query_relevant_llm(query, vectorstore, embeddings, llm_model="gpt-4"):
    """
    Uses an LLM to determine if the user's query is relevant to the content
    potentially covered by the RAG system. It retrieves top documents from the
    vectorstore to provide context for the LLM's judgment.

    Args:
        query (str): The user's question.
        vectorstore (FAISS): The loaded FAISS vector store.
        embeddings (OpenAIEmbeddings): The embeddings model (though not directly used here,
                                       it's passed for consistency if needed for future logic).
        llm_model (str): The OpenAI model to use for the relevance check.

    Returns:
        bool: True if the LLM deems the query relevant, False otherwise.
    """
    if vectorstore is None:
        st.warning("RAG index not found for LLM relevance check. Assuming query is not relevant to RAG content.")
        return False

    try:
        # Retrieve a few top documents to give the LLM context about the RAG's content.
        # This helps the LLM judge relevance based on *what's actually in the RAG*.
        # Using k=5 to provide a broader context for the LLM's judgment.
        retrieved_docs = vectorstore.similarity_search(query, k=5)
        doc_contents = [doc.page_content for doc in retrieved_docs]

        if not doc_contents:
            st.info("No similar documents found in RAG for LLM relevance check. Assuming not relevant to RAG content.")
            return False

        # Combine the retrieved document contents into a single context string
        combined_rag_context = "\n---\n".join(doc_contents)

        # Construct the prompt for the LLM to classify relevance
        prompt = f"""
        Given the following context from a knowledge base, determine if the user's question is relevant to the information contained within this context.
        Respond with 'YES' if the question can likely be answered by or is closely related to the context. Otherwise, respond with 'NO'.

        Context from Knowledge Base:
        {combined_rag_context}

        ---
        User Question: {query}
        ---
        Is the user's question relevant to the provided context? Respond only with 'YES' or 'NO'.
        """

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5, # We expect a short answer like 'YES' or 'NO'
            temperature=0 # Set temperature to 0 for more deterministic output
        )
        llm_decision = response.choices[0].message.content.strip().upper()
        print(f"LLM Relevance Decision: '{llm_decision}' for query: '{query}'") # Log for debugging

        # Check if the LLM's decision contains 'YES'
        if "YES" in llm_decision:
            return True
        else:
            return False

    except openai.APIError as e:
        st.error(f"OpenAI API error during LLM relevance check: {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during LLM relevance check: {e}")
        return False

# --- Streamlit UI ---

def main():
    """
    Main function to run the Streamlit application.
    Sets up the UI for uploading data, collecting it, and asking questions.
    """
    st.set_page_config(page_title="GPT Search Bot with Web RAG", layout="centered")
    st.title("🔎 GPT Search Bot with Web RAG + Selenium")

    st.markdown("""
        This application allows you to:
        1. Upload an Excel file containing URLs to scrape web content.
        2. Build a local Retrieval Augmented Generation (RAG) index from the scraped data.
        3. Ask questions which will first be checked for relevance to the RAG content.
        4. If relevant, it attempts to answer using RAG, falling back to Google Search if RAG fails.
        5. If not relevant, it provides a fixed message indicating no information is available.
    """)

    # --- Data Collection Section ---
    st.header("1. Data Collection (Optional)")
    uploaded_file = st.file_uploader("Upload Excel file with links (first column should contain URLs)", type=['xlsx'])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            # Assuming URLs are in the first column and dropping any empty rows
            urls = df.iloc[:, 0].dropna().tolist()
            st.write(f"Found {len(urls)} URLs in the uploaded file.")
            if st.button("Collect Data from Links and Build RAG Index"):
                with st.spinner("Scraping web content and creating RAG index... This may take a while for many links."):
                    scrape_and_chunk(urls)
                    st.success("✅ Data Collection and RAG Index Creation Completed! You can now ask questions relevant to the collected data.")
        except Exception as e:
            st.error(f"Error processing Excel file: {e}. Please ensure it's a valid .xlsx file with URLs in the first column.")

    # --- Question Answering Section ---
    st.header("2. Ask a Question")
    user_query = st.text_input("Enter your question here:")

    if st.button("Get Answer") and user_query:
        # Basic check for API keys
        if not OPENAI_API_KEY or not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
            st.error("Please ensure all API keys (OPENAI_API_KEY, GOOGLE_API_KEY, SEARCH_ENGINE_ID) are set in your environment variables.")
            return

        with st.spinner("Processing your question..."):
            vectorstore = load_vectorstore()
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) # Needed for vectorstore loading and LLM relevance check

            # --- LLM-BASED RELEVANCE FILTER ---
            is_relevant_to_rag = False
            if vectorstore:
                # Use LLM to check relevance, providing context from RAG
                is_relevant_to_rag = is_query_relevant_llm(user_query, vectorstore, embeddings)
            else:
                # If RAG index is not built, it cannot be relevant to RAG content.
                # This path leads directly to the "no information" message as per requirement.
                st.warning("RAG index not found. Cannot perform relevance check against RAG content. Assuming query is not relevant to RAG.")
                is_relevant_to_rag = False # Explicitly set to False to trigger the 'else' block below

            if is_relevant_to_rag:
                st.info("Query deemed relevant to RAG content by LLM. Attempting RAG answer...")
                rag_answer, rag_sources = ask_rag(user_query)

                # Check if RAG provided a meaningful answer
                if rag_answer and not any(phrase in rag_answer.strip().lower() for phrase in ["i don't know", "idk", "not sure", "no information", "sorry", "i am unable to find"]):
                    st.subheader("Answer from RAG")
                    st.write(rag_answer)
                    if rag_sources:
                        st.subheader("Sources from RAG")
                        for src in rag_sources:
                            st.markdown(f"- [{src}]({src})")
                else:
                    st.warning("RAG could not provide a definitive answer despite LLM relevance check. Falling back to Google Search.")
                    # Fallback to Google Search if RAG answer is not good
                    search_results = google_search(user_query)
                    if search_results:
                        snippets = [item.get('snippet', '') for item in search_results[:5]]
                        links = [item.get('link') for item in search_results[:3]]
                        final_answer = summarize_with_gpt(snippets, user_query)
                        st.subheader("Answer from Google + GPT")
                        st.write(final_answer)
                        st.subheader("Sources from Google Search")
                        for l in links:
                            st.markdown(f"- [{l}]({l})")
                    else:
                        st.error("No answer found from Google Search either. Please try a different query.")
            else:
                # If not relevant to RAG (either by LLM's judgment or because RAG is not built),
                # directly provide the fixed answer and exit
                st.info("The information is not present in the system or is not relevant to the collected RAG content. Please try a different question or upload more relevant data.")
                return # Exit the function as per requirement

if __name__ == '__main__':
    main()