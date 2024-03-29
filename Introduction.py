import streamlit as st
import openai

# Set your OpenAI API key
openai.api_key = st.secrets["API_key"]

def generate_response(question, context):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=question + "\nContext: " + context + "\nQ:",
      temperature=0.7,
      max_tokens=150
    )
    return response.choices[0].text.strip()

def app():
    st.title("OpenAI Text Generation App")
    
    # Text input for user question
    question = st.text_input("Enter your question:")
    
    # Text area input for the context
    context = st.text_area("Enter the context:")
    
    # Button to generate response
    if st.button("Generate Response"):
        if question and context:
            response = generate_response(question, context)
            st.write("Response:")
            st.write(response)
        else:
            st.error("Please enter both question and context.")


#run the app
if __name__ == "__main__":
  app()
