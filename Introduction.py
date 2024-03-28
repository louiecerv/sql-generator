import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Model and tokenizer names (you can choose a different model from Hugging Face)
model_name = "jysh1023/distilbert-base-cased-squad-v2"
tokenizer_name = model_name

access_token = st.secrets["API_key"]
st.write(access_token)

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=access_token)
model = AutoModelForQuestionAnswering.from_pretrained(model_name, token=access_token)

def answer_question(document, question):
  """
  Answers a question about the given document using the loaded model.

  Args:
      document: The text document to be analyzed.
      question: The question to be answered.

  Returns:
      A dictionary containing the answer text, start and end token positions.
  """
  # Encode the document and question using the tokenizer
  inputs = tokenizer(question, document, return_tensors="pt")

  # Perform QA with the model
  outputs = model(**inputs)

  # Get the predicted start and end token positions of the answer
  start_scores, end_scores = outputs.start_logits, outputs.end_logits

  # Decode predicted tokens back to answer text
  answer_start = torch.argmax(start_scores)
  answer_end = torch.argmax(end_scores) + 1
  #answer = tokenizer.convert_tokens_to_strings(inputs["input_ids"][0][answer_start:answer_end])
  answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end], skip_special_tokens=True)

  # Return answer info as a dictionary
  return {"answer": answer[0], "start": answer_start, "end": answer_end}

def app():

  st.title("Document Question Answering")

  # Input fields for document and question
  document = st.text_area("Enter Document Text Here")
  question = st.text_input("Ask a question about the document")

  # Button to trigger question answering
  if st.button("Answer Question"):
    if document and question:
      # Call answer_question function and get answer info
      answer_info = answer_question(document, question)
      answer = answer_info["answer"]

      # Display the answer
      st.write(f"Answer: {answer}")
    else:
      st.warning("Please provide both document text and a question.")

#run the app
if __name__ == "__main__":
  app()
