#1 import streamlit and chatbot file
import streamlit as st 
import  chatbot_backend as demo  #**Import your Chatbot file as demo

#2 Set Title for Chatbot - https://docs.streamlit.io/library/api-reference/text/st.title
st.title("Hi, This is Chatbot Arthur :brain:") # **Modify this based on the title you want in want

# Add "End Conversation button"
end_conversation = st.button("End Conversation")

#3 LangChain memory to the session cache - Session State - https://docs.streamlit.io/library/api-reference/session-state
#t initializes the memory for storing conversation history using the demo_memory() function from the backend.
if 'memory' not in st.session_state: 
    st.session_state.memory = demo.demo_memory() 

#4 Add the UI chat history to the session cache - Session State - https://docs.streamlit.io/library/api-reference/session-state
if 'chat_history' not in st.session_state: #see if the chat history hasn't been created yet
    st.session_state.chat_history = [] #initialize empy list in the session state to store the chat history

#5 Re-render/ shows the chat history (Streamlit re-runs this script, so need this to preserve previous chat messages)
#  with corresponding roles (user or assistant)
for message in st.session_state.chat_history: 
    with st.chat_message(message["role"]): 
        st.markdown(message["text"]) 

#6 Displays a chat input box with the text "Powered by Bedrock and Claude".
input_text = st.chat_input("Powered by Bedrock and Claude") 


# If the uses enters some text
if input_text or end_conversation: 
    if input_text == "/end":  # Check if the user entered the "/end" command
        st.session_state.chat_history = []  # Clear the chat history
        st.session_state.memory = demo.demo_memory()  # Reset the memory
        st.rerun()  # Rerun the app to clear the UI
    elif input_text:    
        # Display the user's message.
        with st.chat_message("user"):  st.markdown(input_text) 
        # Adds user message to chat history
        st.session_state.chat_history.append({"role":"user", "text":input_text}) 
        #gets the chatbot's response using the demo_conversation function from the backend, passing the user's input and the memory. 
        chat_response = demo.demo_conversation(input_text=input_text, memory=st.session_state.memory) #** replace with ConversationChain Method name - call the model through the supporting library
        #display chatbot's response
        with st.chat_message("assistant"): 
            st.markdown(chat_response) 
        # Adds chatbots response to chat_history
        st.session_state.chat_history.append({"role":"assistant", "text":chat_response}) 

    elif end_conversation:  # If the user clicked the "End Conversation" button
        st.session_state.chat_history = []  # Clear the chat history
        st.session_state.memory = demo.demo_memory()  # Reset the memory
        st.rerun() # Force Rerun the app to clear the UI