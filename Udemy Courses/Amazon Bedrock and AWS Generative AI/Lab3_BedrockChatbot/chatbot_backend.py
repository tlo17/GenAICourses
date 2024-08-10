#1 https://python.langchain.com/v0.1/docs/integrations/llms/bedrock/
#pip install -U langchain-aws
#pip install anthropic


#1 import langchain packages
from langchain.chains import ConversationChain # Creates a conversational AI
from langchain.memory import ConversationSummaryBufferMemory #Manage Conversation History
from langchain.prompts import PromptTemplate 
from langchain_aws import ChatBedrock #Allows interation of bedrock service


#2a demo_chatbot creates a ChatBedrock instance, which establishes a connection to the Amazon Bedrock service. 
# It specifies the AWS credentials profile, the model ID, and various inference parameters
def demo_chatbot():
    #Lines 16 - 24 show the API request call and inference parameters 
    demo_llm=ChatBedrock(
       credentials_profile_name='default',
       model_id='anthropic.claude-3-haiku-20240307-v1:0',
       model_kwargs= {
           "max_tokens": 300,
           "temperature": 0.1,
           "top_p": 0.9,
           "stop_sequences": ["\n\nHuman:"]} )
    return demo_llm


# #2b Test the LLM w. using invoke method. Comment out line 24 and lines 30+. Add input_text as parameter in demo_chatbot
# # These lines demonstrate how to use the invoke method of the ChatBedrock instance to generate a response for a given input text. 
#     return demo_llm.invoke(input_text)
# response=demo_chatbot(input_text="Hi, what is your name?")
# print(response)


#3 Demo_memory creates an instance of ConversationSummaryBufferMemory, which is used to store and manage the
# conversation history. It takes the ChatBedrock instance from demo_chatbot and sets a maximum token limit of 300
# this limits how much information from the previous conversation is passed.
def demo_memory():
    llm_data=demo_chatbot()
    memory=ConversationSummaryBufferMemory(llm=llm_data,max_token_limit=300)
    return memory


#4 demo_conversation function creates an instance of ConversationChain, which combines the ChatBedrock 
# instance and the conversation memory. It takes the input text and the memory instance as arguments.
# input_text input provided by user
def demo_conversation(input_text,memory):
    llm_chain_data=demo_chatbot()
    llm_conversation=ConversationChain(llm=llm_chain_data,memory=memory,verbose=True)

#5 Chat response using invoke (Prompt template) the ConversationChain instance with the input text, generating a 
# chat response. It returns the 'response' key from the output dictionary, which contains the generated text.
    chat_reply=llm_conversation.invoke(input_text)
    return chat_reply['response']
