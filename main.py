import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API keys
gemini_key = os.getenv("GEMINI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

# Validate API keys
if not gemini_key:
    raise ValueError("GEMINI_API_KEY is missing in your environment variables.")
if not pinecone_key:
    raise ValueError("PINECONE_API_KEY is missing in your environment variables.")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_key)

# Pinecone index config
INDEX_NAME = "quickstart"
DIMENSION = 1024

# Create index if it doesn't exist

indexes = pc.list_indexes().names()
if INDEX_NAME not in indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )
else:
    print(f"Index '{INDEX_NAME}' already exists. Skipping creation.")

# Connect to the index (assign to variable for later use if needed)
index = pc.Index(INDEX_NAME)
llm = ChatGoogleGenerativeAI(model = "models/gemini-1.5-pro-latest", google_api_key=gemini_key)
# Streamlit UI
st.title("Gen AI Conversation AI Agent")

# Use dynamic user input instead of hardcoded question
user_query = st.text_input("Ask a question about studying abroad:", "Tell me about study in Türkiye")

# Generate and display response
if user_query:
    try:
        response = llm.invoke(user_query)
        st.write(response.content)
    except Exception as e:
        st.error(f"An error occurred while generating response: {e}")
