import streamlit as st
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="federicopascual/finetuning-sentiment-model-3000-samples")
results = classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)

# Print the results
for result in results:
    print(result)

# Config page
st.set_page_config(
    page_title="Model Inference Page",
    page_icon=":compuFter:",
    layout="wide",  # Use "wide" layout for a larger page width
    initial_sidebar_state="expanded",  # Expand the sidebar by default
    )