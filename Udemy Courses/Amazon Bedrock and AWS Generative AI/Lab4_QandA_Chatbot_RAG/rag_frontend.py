import streamlit as st 
import rag_backend as demo 

# Set the page title
st.set_page_config(page_title="Ebay Business Report w RAG")

# Create a custom title with styling
new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Ebay Business Report w RAG 🎯</p>'
st.markdown(new_title, unsafe_allow_html=True)

# Initialize the loading state
if 'backend_initialized' not in st.session_state:
    st.session_state.backend_initialized = False

# Iitializes the backend. Ensures everything is ready before the user starts interacting with the UI
def initialize_backend():
    with st.spinner("🔄 Initializing the backend... Please wait..."):
        # Load and prepare the backend (e.g., creating index, embeddings, etc.)
        demo.hr_index()  # Ensure the index is ready
        st.session_state.backend_initialized = True  # Mark backend as initialized

# Run the backend initialization if not already done
if not st.session_state.backend_initialized:
    initialize_backend()

# Main UI display only after backend is initialized
if st.session_state.backend_initialized:
    # Initialize the chat history if it doesn't exist in the session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Function is triggered when user submits a question. It processes user's question and generates response
    def submit_question():
        # Get the user's input from the text area
        user_question = st.session_state.user_input.strip()
        
        if user_question:  # Check if the question is not empty
            # Add user's question to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Generate and add assistant's response to chat history
            with st.spinner("📢 One moment, generating your answer..."):
                response = demo.hr_rag_response(user_question)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Clear the input box
            st.session_state.user_input = ""

    # Create a text input area for user questions
    st.text_area("Ask a question about the Ebay Business Report:", key="user_input", height=100)

    # Create buttons for submitting questions and ending the conversation
    col1, col2 = st.columns(2)
    with col1:
        st.button("📌 Ask Question", on_click=submit_question, type="primary")
    with col2:
        if st.button("🔚 End Conversation", type="secondary"):
            st.session_state.chat_history = []  # Clear the chat history
            st.rerun()  # Rerun the app to refresh the display

    # Display the chat history with the latest question-response pair first
    if st.session_state.chat_history:
        # Group messages into pairs (question-response)
        paired_history = [
            (st.session_state.chat_history[i], st.session_state.chat_history[i + 1])
            for i in range(0, len(st.session_state.chat_history), 2)
        ]
        
        # Reverse the paired history to display the latest question-response pair first
        for question, response in reversed(paired_history):
            with st.chat_message(question["role"]):
                st.write(question["content"])
            with st.chat_message(response["role"]):
                st.write(response["content"])
    else:
        st.write("No conversation history yet. Start by asking a question.")
