# import dependencies
import streamlit as st
from langchain_community.document_loaders import SeleniumURLLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import nltk

# download tools from NLTK library
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Prompt template for LLM
template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer: 
"""

embeddings = OllamaEmbeddings(model="llama3.2")
vector_store = InMemoryVectorStore(embeddings)

# Instantiate LLM
model = OllamaLLM(model="llama3.2")

# load webpage contents
def load_page(url): 
    loader = SeleniumURLLoader(
        urls=[url],
        headless=False #run in non-headless mode for Cloudflare to see a real browser
    )
    documents = loader.load()
    return documents

# load pdf contents
def load_pdf(uploaded_file): 
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    return PyPDFLoader("temp.pdf").load()

# Split text into smaller documents - that can fit better to LLM context window, for loading into vector database/in-memory vector store. 
def split_text(documents):
    if not documents: 
        return []
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
        chunk_overlap=200, 
        add_start_index=True
    )
    data = text_splitter.split_documents(documents)
    return data

# Index documents with embeddings model and vector store - Split docs would be converted to embeddings (emb model) and stored in vector store.
def index_docs(documents): 
    vector_store.add_documents(documents)

# Retrieve docs - Based on user query.
def retrieve_docs(query):
    return vector_store.similarity_search(query)

# pass context-rich combined prompt to LLM to generate fontext-aware response.
def answer_question(question, context): 
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# Streamlit config? 
st.title("Research Assistant")
url = st.text_input("Enter URL:")
upload = st.file_uploader("Upload a PDF file", type=["pdf"])
documents = []

# load content type - select one.
if url: 
    documents = load_page(url)
elif upload: 
    documents = load_pdf(upload)

# perform pipeline steps
chunked_docs = split_text(documents)
index_docs(chunked_docs)

# use query question to retrieve docs
question = st.chat_input()
if question: 
    st.chat_message("user").write(question) #display user's question.
    retrieve_documents = retrieve_docs(question)
    context = "\n\n".join(doc.page_content for doc in retrieve_documents)
    answer = answer_question(question, context)
    st.chat_message("assistant").write(answer) #display answer.
