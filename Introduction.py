import streamlit as st
import openai


from openai import AsyncOpenAI

def generate_response(question, context):
  client = AsyncOpenAI()
  completion = await client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": question}, {"role": "system", "content": context}])
  return completion.choices[0].message.content


def app():
    client = OpenAI(openai.api_key == st.secrets["API_key"])
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
