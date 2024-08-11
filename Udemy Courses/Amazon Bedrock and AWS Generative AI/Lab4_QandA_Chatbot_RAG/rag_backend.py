import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock
from langchain.chains import RetrievalQA

# Global variable to store the index
global_index = None

# hr_index loads the pdf, splits it into chunks, creates embeddings and builds a searchable index
def hr_index():
    global global_index
    
    # If the index has already been created, return it to avoid rebuilding
    if global_index is not None:
        return global_index

    # Load the PDF document
    data_load = PyPDFLoader('https://www.ebaymainstreet.com/sites/default/files/policy-papers/2021%20Small%20Online%20Business%20Report.pdf')
    
    # Split the Text based on Character, Tokens etc. - Recursively split is by character for processing
    #  ["\n\n", "\n", " ", ""] by Paragraphs, lines, 100 characters  and overlap by 10 characters
    data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100, chunk_overlap=10)
    
    # Create embeddings using Amazon Bedrock
    data_embeddings = BedrockEmbeddings(
        credentials_profile_name='default',
        model_id='amazon.titan-embed-text-v1')
    
    # Create a vector store index
    data_index = VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS)
    
    # Build the index from the loaded document
    global_index = data_index.from_loaders([data_load])
    return global_index

# Create and configure the language language model (claude) by leveraging Amazon Bedrock API.
def hr_llm():
    return Bedrock(
        credentials_profile_name='default',
        model_id='anthropic.claude-v2',
        model_kwargs={
            "max_tokens_to_sample": 3000,
            "temperature": 0.1,
            "top_p": 0.9
        })

#Generate question responses using Retrieval-Augmented Generation (RAG). Creates embeddings, does similarity search to identify best response.
# Args:  question (str): The user's question. Returns: Generated response 
def hr_rag_response(question):
    # Get or create the index
    index = hr_index()
    # Get the language model
    rag_llm = hr_llm()
    
    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=rag_llm,
        chain_type="stuff",
        retriever=index.vectorstore.as_retriever(),
        return_source_documents=True
    )
    # Get the response
    response = qa_chain({"query": question})
    return response['result']