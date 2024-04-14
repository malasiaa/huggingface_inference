import streamlit as st
from transformers import pipeline
import json

# URL of the Hugging Face favicon
hugging_face_favicon_url = "https://huggingface.co/favicon.ico"

# Config page
st.set_page_config(
    page_title="Inference Page",
    page_icon=hugging_face_favicon_url,
    #layout="wide",  # Use "wide" layout for a larger page width
    initial_sidebar_state="expanded",  # Expand the sidebar by default
    )

# Add Hugging Face icon in the title
st.markdown(
    f"""
    <h1 style="display: flex; align-items: center;">
        Inference Page <img src="{hugging_face_favicon_url}" style="height: 50px; margin-right: 10px;">
    </h1>
    """,
    unsafe_allow_html=True,
)
#Selectbox to choose the type of task
task_type = st.selectbox('Select Task:', ['Sentiment Analysis', 'Text Generation'])
default_message = "Please insert your sentence HERE..."

# Add a text input space
user_input = st.text_input("Enter your text here:", default_message )



# You can then use the user_input variable in your code
# For example, to display the input text
if user_input == default_message and task_type == 'Sentiment Analysis':
    st.markdown("""
    <span style="font-size: 15px; color: #D3D3D1;">
    For multiple separate them with backslash ej: Rice is good. \\ Rice is disgusting. \\ I don't like pasta.
    </span>
    """, unsafe_allow_html=True)

if user_input != default_message:
    if task_type == 'Sentiment Analysis':
        sentences = user_input.split('\\ ')
        classifier = pipeline("sentiment-analysis", model="federicopascual/finetuning-sentiment-model-3000-samples")
        results = classifier(sentences)
        # Output format is a dict [{'label': 'LABEL_1', 'score': 0.71...}]
        
        sentence = 0
        # Process the results (this part depends on how you want to display the results)
        for result in results:
            
            # Extract the label and score
            label = result['label']
            score = result['score']

            # Determine the message based on the label
            if label == 'LABEL_1':
                message = "Positive"
            elif label == 'LABEL_0':
                message = "Negative"
            else:
                message = "Unknown"
            st.write(sentences[sentence])
            # Display the message and score in two lines with "Classification" in bold
            st.write(f"**Sentiment**: {message}.     **Score**: {score}")
            sentence += 1

    if task_type == 'Text Generation':
        classifier = pipeline("text-generation", model="gpt2")

        # max_length: Controls the maximum length of the generated text.
        # num_return_sequences: Specifies the number of generated sequences.
        # temperature: Controls the randomness of the output. Lower values make the output more deterministic.
        # Adjusting temperature to make the output more focused.
        # Increasing top_k to reduce randomness.
        # Adjusting top_p to control the cumulative probability.
        # Setting no_repeat_ngram_size to avoid repetition.
        results = classifier([user_input], max_length=150, num_return_sequences=1, temperature=0.3, top_k=50, top_p=0.95, no_repeat_ngram_size=1)

        generated_text = results[0][0]['generated_text']
        st.write(f"**Text**: {generated_text}...")

