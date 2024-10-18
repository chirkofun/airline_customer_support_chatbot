import os
from dotenv import load_dotenv
from langchain_together import ChatTogether

from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

load_dotenv(override=True)

llm = ChatTogether(api_key=os.getenv("TOGETHER_API_KEY"), temperature=0.0, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

# Create an embedding
embedding = HuggingFaceEmbeddings()

db = FAISS.load_local("airline_db", embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever
)

# Test server route
@app.route("/ping", methods=['GET'])
def pinger():
    return "<p>Hello world!</p>"

# Main route to handle POST requests
@app.route("/bot", methods=['POST'])
def bot():
    query = request.form.get("Body")
    response = qa_chain.run(query)
    resp = MessagingResponse()
    msg = resp.message()
    msg.body(response)
    return str(resp)

if __name__ == '__main__':
    app.run(port = 8888)

