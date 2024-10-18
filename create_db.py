from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    PyPDFLoader("airlines.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split the docs into chunks and store it in splits
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800, # max size of each chunk in num characters
    chunk_overlap = 200, # overlap between chunk in num characters
    separators=["\n\n", "\n", ".", " "]
)
splits = text_splitter.split_documents(docs)

# Create embeddings
from langchain.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings()

# Create db (vector store) from the document
from langchain_community.vectorstores import FAISS
db = FAISS.from_documents(splits, embedding)

# Save db locally
db.save_local("airline_db")