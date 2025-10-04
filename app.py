import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# ------------------------------
# Load Model & Tokenizer
# ------------------------------
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("flirty_model")
    tokenizer = BertTokenizer.from_pretrained("flirty_model")
    return model, tokenizer

model, tokenizer = load_model()

# ------------------------------
# Chatbot Function
# ------------------------------
def get_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=150)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    if prediction == 1:
        return "😉 That sounds flirty!"
    else:
        return "😊 Just a normal message."

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Flirty Chatbot", page_icon="💬")
st.title("💬 Flirty Chatbot (BERT)")

user_input = st.text_input("Type your message:")

if st.button("Check"):
    if user_input.strip():
        response = get_response(user_input)
        st.write("🤖 Bot:", response)
