import streamlit
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="federicopascual/finetuning-sentiment-model-3000-samples")
results = classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)

# Print the results
for result in results:
    print(result)
