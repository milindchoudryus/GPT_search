# GPT-Powered Google Search Chatbot

This is a Streamlit app that uses Google Custom Search API to pull live search results and summarizes them using OpenAI GPT-4.

## Features

- Enter any question
- Fetch top Google search results
- Summarize into a concise answer using GPT-4
- Display answer and clickable source links

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/milindchoudryus/GPT_search.git
   cd GPT_search
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key
   SEARCH_ENGINE_ID=your_search_engine_id
   OPENAI_API_KEY=your_openai_api_key
   ```

4. Run the app:
   ```
   streamlit run app.py
   ```

## Notes

- Requires valid Google Custom Search Engine and OpenAI API credentials.
- Designed for local use; for deployment, configure `.env` securely.
