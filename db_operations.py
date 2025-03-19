from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#Load raw pdf
DATA_DIR = 'files/'
def load_files_from_directory(directory):
    loader = DirectoryLoader(directory, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_files_from_directory(directory=DATA_DIR)

print("length of pages: ", len(documents))

#Create chunks
def create_chunks(document):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(document)   
    return chunks

chunks = create_chunks(document=documents)
print("length of chunks: ", len(chunks))

#Create vector embeddings
def get_embeddings_model():
    embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings_model

embeddings_model = get_embeddings_model()

# Store embeddings in FAISS
DB_PATH = 'db/db_faiss'
db = FAISS.from_documents(chunks, embeddings_model)
db.save_local(DB_PATH)