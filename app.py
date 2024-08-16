import os
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up Streamlit app configuration
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Retrieve Groq API Key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.sidebar.error("GROQ_API_KEY is not set. Please set it in your .env file.")

# Input for the URL to be summarized
generic_url = st.text_input("URL", label_visibility="collapsed")

# Initialize the LLM using the Groq API Key
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

# Define the prompt template for summarization
prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Summarization process when the button is clicked
if st.button("Summarize the Content from YT or Website"):
    # Validate all the inputs
    if not groq_api_key or not generic_url.strip():
        st.error("Please provide the necessary information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or a website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                # Load content from the provided URL (YouTube or website)
                if "youtube.com" in generic_url:
                    try:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                        docs = loader.load()
                    except Exception as e:
                        st.error(f"Error loading YouTube video: {e}")
                        st.stop()
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                    docs = loader.load()

                # Create a summarization chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
