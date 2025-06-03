#importing the chat llm
from langchain_groq import ChatGroq
#imports for the Rag
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain.chains import create_retrieval_chain,create_history_aware_retriever,load_summarize_chain,LLMMath_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
#imports for the search_engine
from langchain.agents import initialize_agent,AgentType
from langchain.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain.tools import WikipediaQueryRun,ArxivQueryRun,DuckDuckGoSearchRun
#imports  for conversational q_a Rag bot
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
st.set_page_config(page_title='AI World',page_icon=':reminder_ribbon:')
st.title('Welcome to AI suite')
st.subheader('Created by Narendra AI Technologies')

groq_api=st.sidebar.text_input('Enter the GROQ_API',type='password')
if  not groq_api:
    st.info('Please Enter the API')
    st.stop()
groq_llm=ChatGroq(model='llama-3.1-8b-instant',api_key=groq_api)

if 'messages' not in st.session_state:
    st.session_state['messages']={}
def get_session_store(session_id:str) :
    if session_id not in st.session_state.messages:
        st.session_state.messages[session_id]=ChatMessageHistory()
    return st.session_state.messages[session_id]
file_=st.sidebar.file_uploader('hey upload you Pdf',type='pdf')
#RAG TOOL
def rag_chain(query:str,session_id:str):
    st.session_state.loader=PyPDFLoader('attention.pdf')
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.rcts=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.chunks=st.session_state.rcts.split_documents( st.session_state.docs)
    st.session_state.embedder=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",)
    db=Chroma.from_documents(documents=st.session_state.chunks,embedding=st.session_state.embedder)
    reteriver=db.as_retriever()
    template_='''Answer the queries using the help of the below context /n/n
            <context>
            {context}
            </context>
          
    '''
    contexulize_prompt='''Act as a helper agent ,and contexualize  the relevant question from the history provided to you within the less words,
    if you don't know or get from the history leave it  
    '''
    hepler_ai_prompt=ChatPromptTemplate.from_messages(
        [('system',contexulize_prompt),
         MessagesPlaceholder('messages'),
         ('user','{input}')
         

        ]
    )
    history_aware_chain=create_history_aware_retriever(llm=groq_llm,retriever=reteriver,prompt=hepler_ai_prompt)
    prompt=ChatPromptTemplate.from_messages([
        ('system',template_),
        MessagesPlaceholder('messages'),
        ('user','{input}')
    ])
    doc_chain=create_stuff_documents_chain(llm=groq_llm,prompt=prompt)
    reterival_chain=create_retrieval_chain(history_aware_chain,doc_chain)
    model=RunnableWithMessageHistory(reterival_chain,get_session_store,input_messages_key='input',history_messages_key='messages')
    
    response=model.invoke({'input':query},config={'configurable':{'session_id':session_id}})
    return response['answer']

def rag_tool(query_of:str):
    return rag_chain(query=query_of,session_id='research')

from langchain.tools import Tool
rag_tools=Tool(
    name='contextuailize Research Tool',
    func=rag_tool,
    description='It is a specilaized RAG tool which is equipped with helper tool for rememebering the history and' \
    'takes the context from the uploaded pdf and answer the query'

)
#preapring the tools like wiki,arxiv,duckduck
math_chain=LLMMath_chain.from_llm(groq_llm)
math_tool=Tool(
   name='calculator_tool',
   func=math_chain.run,
   description='A tools for answering math related questions. Only input mathematical expression need to bed provided'
)

   
   
wikiwrapper=WikipediaAPIWrapper(top_k_results=2)
arxiv=ArxivAPIWrapper(top_k_results=2)
wiki_tool=WikipediaQueryRun(api_wrapper=wikiwrapper)
arxiv_tool=ArxivQueryRun(api_wrapper=arxiv)
search_tool=DuckDuckGoSearchRun()
#summarizer of youtube urls  and other url
import validators
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader

def summarizer(url:str):
   prompt_temp='''Based on the context given please summarize the para with neat headings and highlight necessary
                         context:{context}'''
   prompt=PromptTemplate.from_template(template=prompt_temp
,input_variables=['context'])
   if not validators.url(url.strip()):
     print('Not valid url')
   else:
     if 'youtube.com' in url.strip():
       loader=YoutubeLoader.from_youtube_url(url,add_video_info=True)
     else:
       loader=UnstructuredURLLoader(urls=[url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
       
       docs=loader.load()
       chain=load_summarize_chain(docs,prompt)
       output_summary=chain.run(docs)

       return output_summary

summarizer_tool=Tool(
  name='summarizer tool',
  func=summarizer,
  description='It is summarizer tool uses load and summarize chain .Loads the data from the urls provided and can identify Wrong urls also' \
  'can work with youtube and other urls to extract and summarize the data'

) 

   
tools=[rag_tools,summarizer_tool,wiki_tool,arxiv_tool,search_tool]


user_prompt=st.text_input('Enter you prompt')

if user_prompt:
   sb_c=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
   final_agent=initialize_agent(
    tools=tools,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,llm=groq_llm,handle_parsing_errors=True)
   response=final_agent.run(user_prompt,callbacks=[sb_c] )
   st.success(response)
   

   



