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

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# Global storage
INDEX_PATH = "rag_index"

# Google Search Fallback
def google_search(query):
    params = {'q': query, 'key': GOOGLE_API_KEY, 'cx': SEARCH_ENGINE_ID}
    response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
    if response.status_code != 200:
        print("❌ Google Search API failed:", response.text)
        return None
    return response.json().get('items', [])

def summarize_with_gpt(snippets, user_query):
    prompt = f"Using the following search results, answer the user's question:\n\n{snippets}\n\nQuestion: {user_query}\nAnswer:"
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

# Selenium Scraper
def selenium_scrape(url):
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(5)  # Allow time for JS to render
        text = driver.find_element(By.TAG_NAME, "body").text
        driver.quit()
        return text
    except Exception as e:
        print(f"❌ Selenium scraping failed for {url}: {e}")
        return ""

# Data Collection from URLs
def scrape_and_chunk(urls):
    all_chunks = []
    for url in urls:
        print(f"🔗 Scraping: {url}")
        text = selenium_scrape(url)
        if len(text) < 300:
            print(f"⚠️ Skipped {url}: too little text ({len(text)} characters)")
            continue
        print(f"✅ Scraped {len(text)} characters from: {url}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.create_documents([text])
        for doc in docs:
            doc.metadata = {"source": url}
        all_chunks.extend(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)

# Load RAG
def load_vectorstore():
    if not os.path.exists(INDEX_PATH):
        return None
    return FAISS.load_local(
        INDEX_PATH,
        OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        allow_dangerous_deserialization=True
    )

def ask_rag(question):
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return None, None
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=OPENAI_API_KEY),
        retriever=vectorstore.as_retriever()
    )
    response = qa_chain({"query": question})
    answer = response['result']
    sources = [doc.metadata['source'] for doc in vectorstore.similarity_search(question, k=3)]
    return answer, sources

# UI
def main():
    st.title("🔎 GPT Search Bot with Web RAG + Selenium")

    # Upload Excel
    uploaded_file = st.file_uploader("Upload Excel file with links", type=['xlsx'])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        urls = df.iloc[:, 0].dropna().tolist()
        if st.button("Collect Data from Links"):
            with st.spinner("Scraping & Creating RAG..."):
                scrape_and_chunk(urls)
                st.success("✅ Data Collection Completed")

    # Ask a question
    user_query = st.text_input("Ask a question")
    if st.button("Get Answer") and user_query:
        with st.spinner("Thinking..."):
            rag_answer, rag_sources = ask_rag(user_query)
            if rag_answer and rag_answer.strip().lower() not in ["i don't know.", "idk", "not sure", ""]:
                st.subheader("Answer from RAG")
                st.write(rag_answer)
                st.subheader("Sources")
                for src in rag_sources:
                    st.write(src)
            else:
                search_results = google_search(user_query)
                if search_results:
                    snippets = "\n".join([item.get('snippet', '') for item in search_results[:5]])
                    links = [item.get('link') for item in search_results[:3]]
                    final_answer = summarize_with_gpt(snippets, user_query)
                    st.subheader("Answer from Google + GPT")
                    st.write(final_answer)
                    st.subheader("Sources")
                    for l in links:
                        st.write(l)
                else:
                    st.error("No answer found. Try again.")

if __name__ == '__main__':
    main()
