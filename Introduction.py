import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from transformers import pipeline, Trainer, TrainingArguments


# Model and tokenizer names (you can choose a different model from Hugging Face)
model_name = "google/gemma-7b-it"
tokenizer_name = model_name

access_token = st.secrets["API_key"]

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=access_token)
model = AutoModelForQuestionAnswering.from_pretrained(model_name, token=access_token)

# Function to preprocess text data (add your specific preprocessing steps here)
def preprocess_text(text):
  # Example: Remove punctuation and lowercase text
  text = text.lower()
  text = "".join([char for char in text if char.isalnum() or char == " "])
  return text

def train_gemma(document):
  """Trains a Gemma model on a provided document"""
  # Preprocess document text
  preprocessed_text = preprocess_text(document)

  # Define training data
  train_data = [{"input_ids": tokenizer(preprocessed_text, return_tensors="pt", token=access_token)["input_ids"],}]

  # Define tokenizer and model
  tokenizer = pipeline("text-generation").tokenizer
  model_name = "google/gemma-2b-it"

  # Define training arguments (adjust parameters as needed)
  training_args = TrainingArguments(
      output_dir="./gemma_model",
      overwrite_output_dir=True,
      per_device_train_batch_size=4,
      save_steps=10_000,
      save_total_limit=2,
      token=access_token
  )

  # Create a trainer instance
  trainer = Trainer(
      model=model_name,
      args=training_args,
      train_dataset=train_data,
      token=access_token
  )

  # Train the model
  trainer.train()
  return pipeline("text-generation", model="./gemma_model")

# Initialize variables
document = ""
gemma = None
chat_history = []

def update_chat(user_input):
  """Updates chat history and returns Gemma's response"""
  if not gemma:
    st.error("Please upload a document and train the model first.")
    return

  chat_history.append({"user": user_input})
  response = gemma(user_input, max_length=1000, num_return_sequences=1)[0]["generated_text"]
  chat_history.append({"gemma": response})
  return response

def app():
  st.title("Chat with Gemma on your Document")

  # File upload for document
  uploaded_file = st.file_uploader("Upload Document (Text only):")

  if uploaded_file:
    document = uploaded_file.read().decode("utf-8")
    st.success("Document uploaded!")

  # Train button
  if st.button("Train Gemma"):
    if document:
      gemma = train_gemma(document)
      st.success("Gemma trained successfully!")
    else:
      st.warning("Please upload a document before training.")

  # Text input field for user message
  user_input = st.text_input("You:")

  # Button to trigger chat update
  if st.button("Send"):
    if user_input:
      response = update_chat(user_input)
      st.write("Gemma:", response)
    else:
      st.warning("Please enter a message to chat with Gemma.")

  # Display chat history (if Gemma is trained)
  if gemma:
    for message in chat_history:
      if message["user"]:
        st.write("You:", message["user"])
      else:
        st.write("Gemma:", message["gemma"])

#run the app
if __name__ == "__main__":
  app()
