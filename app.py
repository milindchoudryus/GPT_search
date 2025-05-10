import os
import requests
import openai
import streamlit as st
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get keys from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client (v1.x syntax)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def google_search(query):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': GOOGLE_API_KEY,
        'cx': SEARCH_ENGINE_ID
    }
    response = requests.get(search_url, params=params)
    if response.status_code != 200:
        print(f"Google API error: {response.status_code}, {response.text}")
        return None
    data = response.json()
    items = data.get('items', [])
    print(f"Retrieved {len(items)} search results.")
    return items

def summarize_with_gpt(snippets, user_query):
    combined_snippets = "\n".join(snippets)
    prompt = f"Using the following search results, answer the user's question:\n\n{combined_snippets}\n\nUser Question: {user_query}\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def main():
    st.title("🔍 GPT-Powered Chatbot")

    user_query = st.text_input("Enter your question:")
    if st.button("Get Answer") and user_query:
        with st.spinner("Searching Google and summarizing with GPT..."):
            search_results = google_search(user_query)
            if search_results is None:
                st.error("Search API request failed.")
                return
            if len(search_results) == 0:
                st.error("No search results found.")
                return

            snippets = [item.get('snippet', '') for item in search_results[:5]]
            sources = [item.get('link') for item in search_results[:3]]

            final_answer = summarize_with_gpt(snippets, user_query)

            st.subheader("Answer")
            st.write(final_answer)

            st.subheader("Sources")
            for src in sources:
                st.write(src)

if __name__ == "__main__":
    main()
