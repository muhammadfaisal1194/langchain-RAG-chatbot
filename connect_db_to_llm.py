import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

#Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.environ.get('HF_TOKEN')
HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(repo_id = huggingface_repo_id,
    temperature = 0.5,
    model_kwargs = { "token": HF_TOKEN, "max_length": 512})
    return llm

#Connect LLM with FAISS and create a QA chain

CUSTOM_PROMPT = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answers, just say that you dont know, dont try to make up an answer.
Dont provide any information that is not present in the context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt_template(custom_prompt):
    prompt = PromptTemplate(template = custom_prompt, input_variables = ['context', 'question'])
    return prompt

#Load database
DB_PATH = 'db/db_faiss'
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)

#Create a QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGING_FACE_REPO_ID),
    chain_type='stuff',
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt_template(CUSTOM_PROMPT)}
)

#Ask a question
question = input("Write your question here: ")
response = qa_chain.invoke({"query": question})
print("Answer: ", response['result'])
print("Source documents: ", response['source_documents'])