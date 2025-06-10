# Chainlit app.py

import os
import openai
import chainlit as cl

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager
from llama_index.core.service_context import ServiceContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

openai.api_key = os.environ.get("OPENAI_API_KEY")

try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    index = load_index_from_storage(storage_context)
except:
    documents = SimpleDirectoryReader("RAG-webscraper/docs").load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist()

# Called when new chat session is created
@cl.on_chat_start
async def start():
    Settings.llm = OpenAI(
        model="gpt-4o-mini", temperature=0.1, max_tokens=1024, streaming=True
    )
    Settings.embed_model = Settings.embed_model = HuggingFaceEmbedding(model_name="hkunlp/instructor-large")
    Settings.context_window = 4096

    Settings.callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])
    service_context = Settings.callback_manager

    # how to incorporate memory to query engine? Use CHAT ENGINE instead of query engine!
    chat_engine = index.as_chat_engine(
        streaming=True, similarity_top_k=6, service_context=service_context
    )
    cl.user_session.set("chat_engine", chat_engine)

    await cl.Message(
        author="Assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    chat_engine = cl.user_session.get("chat_engine")

    msg = cl.Message(content="", author="Assistant")

    res = await cl.make_async(chat_engine.stream_chat)(message.content)

    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()