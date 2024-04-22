import streamlit as st
import os
from transformers import pipeline, Conversation

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

# Check if the HF_HOME environment variable is set
hf_home = os.getenv('HF_HOME')

# If HF_HOME is set, use it as the base for the cache directory
if hf_home:
    cache_directory = os.path.join(hf_home, 'hub')
else:
    # If HF_HOME is not set, use the default location
    cache_directory = os.path.expanduser('~/.cache/huggingface/hub')

#function to verify if model weights are already in cache
def check_model_cache (model):
    if model in os.listdir(cache_directory):
        pass
    else:
        st.markdown("""
                    <span style="font-size: 15px; color: #D3D3D1;">
                    Model run for the first time.

                    Downloading model weights to cache.....
                    </span>
                    """, unsafe_allow_html=True)
    
    
#Selectbox to choose the type of task
task_type = st.selectbox('Select Task:', ['Sentiment Analysis', 'Text Generation', 'Summarization', 'Conversational'])
default_message = "Please insert your sentence HERE..."

# Add a text input space
user_input = st.text_input("Enter your text here:", default_message )

#creating a global variable to receive the model role for conversational tasks
role = []
if task_type == 'Conversational':
    role.append(st.text_input("What is the model role?", "Ex: You are a kind pirate."))

# equal to default message or "", will appear the below
if user_input == default_message or user_input == "":
        if task_type == 'Sentiment Analysis':
            st.markdown("""
                <span style="font-size: 15px; color: #D3D3D1;">
                For multiple please separate with backslash ej: Rice is good. \\ Rice is disgusting. \\ I don't like pasta.
                </span>
                """, unsafe_allow_html=True)
        if task_type == 'Text Generation':
            st.markdown("""
                <span style="font-size: 15px; color: #D3D3D1;">
                Powered by ramblings of LLMs
                </span>
                """, unsafe_allow_html=True)
        if task_type == 'Conversational':
            st.markdown("""
                <span style="font-size: 15px; color: #D3D3D1;">
                Powered by ramblings of LLMs
                </span>
                """, unsafe_allow_html=True)

# when there's an input to be runned 
if user_input != default_message:
    if  user_input != "":
        if task_type == 'Sentiment Analysis':
            sentences = user_input.split('\\ ')
            check_model_cache("models--federicopascual--finetuning-sentiment-model-3000-samples")
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

        if task_type == 'Text Generation':
            check_model_cache("models--gpt2")
            classifier = pipeline("text-generation", model="gpt2", temperature=0.9, min_new_tokens=20, max_new_tokens=150)

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
            check_model_cache ("models--microsoft--DialoGPT-medium")
            classifier = pipeline("summarization", model="microsoft/DialoGPT-medium")
            results = classifier([user_input], min_new_tokens=20, max_new_tokens=560, temperature=0.5)
            st.write(f"**Here's the summary**: {results[0]['summary_text']}...")

        if task_type == 'Conversational':
            check_model_cache ("models--microsoft--DialoGPT-medium")
            classifier = pipeline("conversational", model="microsoft/DialoGPT-medium")
            conversation = [
                {"role": "system", "content": role},
                {"role": "user", "content": user_input}
            ]
            results = classifier(conversation, padding=True, truncation=True, return_tensors="pt", padding_side='left')
            last_response = results[-1]['content']
            st.write(f"**Answer**: {last_response}")
    