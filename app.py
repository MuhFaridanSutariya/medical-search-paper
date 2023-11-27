import os
import random
import string
import tempfile

# from apikey import openapikey, serpapikey
from langchain.llms import OpenAI
from pdfloader import loadPDF
from pdfloader import queryPDF

# For utilizing agent search tools
from langchain.agents import load_tools
from langchain.agents import initialize_agent

# For chaining responses
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory

# For frontend user interfacing with app
import streamlit as st
from streamlit_chat import message

openapikey = ""
serpapikey = ""

@st.cache_resource
def initialize():
    os.environ['OPENAI_API_KEY'] = openapikey
    os.environ['SERPAPI_API_KEY'] = serpapikey    

    # temprature 0.0 for consistency response for patients.
    llm = OpenAI(temperature = 0.0)

    # Streamlit is breaking the memory usage because it keeps re-declaring the conversation buffer memory
    tool_names =  ["arxiv", "serpapi"]
    tools = load_tools(tool_names)
    agent_memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(tools, llm, agent="conversational-react-description", verbose=True, memory=agent_memory)

    return agent

def main():
    agent = initialize()
    st.title('🦜 Search and query academic medical papers ')
    
    # Dropdown to select between features
    selected_feature = st.selectbox("Select Feature", ["Question Answering", "PDF Query"])

    if selected_feature == "Question Answering":
        prompt = st.text_input("What medical topic would you like to know about? ")      

        if prompt:
            if 'answer' not in st.session_state:
                st.session_state['answer'] = []
            if 'question' not in st.session_state:
                st.session_state['question'] = [] 

            response = agent.run(prompt)

            st.session_state.question.insert(0, prompt)
            st.session_state.answer.insert(0, response)   

            # Display the chat history
            for i in range(len(st.session_state.question)):        
                questionKey = ''.join(random.choice(string.ascii_letters) for i in range(10))
                answerKey = ''.join(random.choice(string.ascii_letters) for i in range(10))
                
                message(st.session_state['question'][i], is_user=True, key=questionKey)
                message(st.session_state['answer'][i], is_user=False, key=answerKey)
    else:
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

        if uploaded_file:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            pdf_path = temp_file.name

            docsearch = loadPDF(pdf_path)

            query = st.text_input("What medical topic would you like to know about from the PDF?")

            if query:
                pdf_response = queryPDF(query, docsearch)
                st.write("Response from the PDF:", pdf_response)

            temp_file.close()
            os.unlink(temp_file.name)

if __name__ == '__main__':
    main()