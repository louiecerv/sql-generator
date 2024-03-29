import streamlit as st
import openai

# Set your OpenAI API key
openai.api_key = st.secrets["API_key"]

def generate_response(question, context):
    openai.organization = "West Visayas State University"  # Set your OpenAI organization if applicable
    model_engine = "text-davinci-4"  # Use the latest model engine

    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[
            {
                "role": "system",
                "content": "Question: " + question,
            },
            {
                "role": "system",
                "content": "Context: " + context,
            }
        ],
        max_tokens=150,
        n=1,
        stop=None,  # Allow for longer responses if needed
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

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
