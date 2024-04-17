import streamlit as st
from transformers import pipeline
import json

# URL of the Hugging Face favicon
hugging_face_favicon_url = "https://huggingface.co/front/assets/homepage/hugs-mobile.svg"

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
        Inference Page <img src="{hugging_face_favicon_url}" style="height: 80px; margin-right: 10px;">
    </h1>
    """,
    unsafe_allow_html=True,
)
#Selectbox to choose the type of task
task_type = st.selectbox('Select Task:', ['Sentiment Analysis', 'Text Generation', 'Zero-shot classification', 'Summarization', 'Question Answering'])
default_message = "Please insert your sentence HERE..."

# Add a text input space
user_input = st.text_input("Enter your text here:", default_message )


# You can then use the user_input variable in your code
# For example, to display the input text
if user_input == default_message or user_input == "":
        if task_type == 'Sentiment Analysis':
            st.markdown("""
                <span style="font-size: 15px; color: #D3D3D1;">
                For multiple separate them with backslash ej: Rice is good. \\ Rice is disgusting. \\ I don't like pasta.
                </span>
                """, unsafe_allow_html=True)
        if task_type == 'Text Generation':
            st.markdown("""
                <span style="font-size: 15px; color: #D3D3D1;">
                Powered by ramblings of LLMs
                </span>
                """, unsafe_allow_html=True)
        if task_type == 'Summarization':
            st.markdown("""
                <span style="font-size: 15px; color: #D3D3D1;">
                Maximum output tokens: 300 
                            
                Powered by facebook/bart
                </span>
                """, unsafe_allow_html=True)
        if task_type == 'Question Answering':
            context = st.text_input("Enter your context:", default_message )



if user_input != default_message:
    if  user_input != "":
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
                if len(results) >= 2:
                    st.write(sentences[sentence])
                # Display the message and score in two lines with "Classification" in bold
                st.write(f"**Sentiment**: {message}.     **Score**: {score}")
                sentence += 1

        if task_type == 'Zero-shot classification':
            classifier = pipeline("zero-shot-classification", model="federicopascual/finetuning-sentiment-model-3000-samples")
            results = classifier(user_input, candidate_labels=["education", "politics", "meteorology"])
            st.write(f"**Category**: {results}.     **Score**:")

        if task_type == 'Text Generation':
            classifier = pipeline("text-generation", model="gpt2", temperature=0.9, max_new_tokens=100)

            # max_length: Controls the maximum length of the generated text.
            # num_return_sequences: Specifies the number of generated sequences.
            # temperature: Controls the randomness of the output. Lower values make the output more deterministic.
            # Adjusting temperature to make the output more focused.
            # Increasing top_k to reduce randomness.
            # Adjusting top_p to control the cumulative probability.
            # Setting no_repeat_ngram_size to avoid repetition.
            results = classifier([user_input])
            generated_text = results[0][0]['generated_text']
            st.write(f"**Text**: {generated_text}...")

        if task_type == 'Summarization':

            classifier = pipeline("summarization", model="facebook/bart-large-cnn")
            results = classifier([user_input], min_new_tokens=160, max_new_tokens=560, temperature=0.7)
            st.write(f"**Here's the summary**: {results[0]['summary_text']}...")

        if task_type == 'Question Answering':

            classifier = pipeline("question-answering", model="deepset/roberta-base-squad2")
            context = st.text_input("Enter your context:", default_message
            results = classifier(question=user_input, context = context)
            st.write(f"**Answer**: {results}...")