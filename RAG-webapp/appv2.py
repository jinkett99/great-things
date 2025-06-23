# Chainlit app.py

import os
import openai
import chainlit as cl
from typing import Optional
from asyncio.log import logger
from fastapi import Request, Response
from chainlit.types import ThreadDict

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core.callbacks import CallbackManager
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import PromptTemplate

openai.api_key = os.environ.get("OPENAI_API_KEY")
Settings.llm = OpenAI(
    model="gpt-4o-mini", temperature=0.1, max_tokens=1024, streaming=True
)
Settings.embed_model = Settings.embed_model = HuggingFaceEmbedding(model_name="hkunlp/instructor-large")
Settings.context_window = 4096

# Set-up llama guard 3 with Ollama
Settings.gllm = Ollama(
    model="llama-guard3",
    request_timeout=120.0,
)

try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    index = load_index_from_storage(storage_context)
except:
    documents = SimpleDirectoryReader("docs").load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model)
    index.storage_context.persist()

# Integrate new indices with storage context. 

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """Password auth handler for login"""
    
    if (username, password) == ("admin", "admin"): #For development
        return cl.User(identifier="admin", metadata={"role": "ADMIN"})
    else: 
        return None

# Called when new chat session is created
@cl.on_chat_start
async def start():
    '''Handler for chat start events. Set session variables: Chat Engine.'''
    # Now we can see who's starting the conversation!
    user = cl.user_session.get("user")
    logger.info(f"{user.identifier} has started the conversation")

    # set callback handler to enable Chainlit to display intermediate steps in the UI
    Settings.callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])
    service_context = Settings.callback_manager

    # define & set session memory buffer
    memory = ChatMemoryBuffer.from_defaults()
    cl.user_session.set("memory", memory)

    # define chat engine
    chat_engine = index.as_chat_engine(chat_mode="condense_question", llm=Settings.llm, streaming=True, verbose=True, service_context=service_context)

    # Set chat_engine for user session
    cl.user_session.set("chat_engine", chat_engine)

    await cl.Message(
        author="Assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    '''On message handler to handle message received events.'''

    # get session variables
    memory = cl.user_session.get("memory")
    chat_history = memory.get()
    chat_engine = cl.user_session.get("chat_engine")
    msg = cl.Message(content="", author="Assistant")

    # Define moderation function for moderating query and response
    def guardrail_moderate(query, chat_history):
        # Moderate the user input
        moderator_response_for_input = Settings.gllm.complete(query).text
        print(f"moderator response for input: {moderator_response_for_input}")

        # Check if the moderator response for input is safe
        if moderator_response_for_input == "safe":
            response = chat_engine.stream_chat(query, chat_history=chat_history)

            # Moderate the LLM output
            moderator_response_for_output = Settings.gllm.complete(str(response)).text
            print(
                f"moderator response for output: {moderator_response_for_output}"
            )

            # Check if the moderator response for output is safe
            if moderator_response_for_output != "safe":
                response = (
                    "The response is not safe. Please ask a different question."
                )
        else:
            response = "This query is not safe. Please ask a different question."

        return response

    # stream response - CHECK!
    res = await guardrail_moderate(message.content, chat_history)

    for token in res.response_gen:
        await msg.stream_token(token)

    # Update memory buffer
    memory.put(
        ChatMessage(
            role = MessageRole.USER,
            content= message.content
        )
    )
    memory.put(
        ChatMessage(
            role = MessageRole.ASSISTANT,
            content = str(response)
        )
    )
    cl.user_session.set("memory", memory)

    await msg.send()

@cl.on_stop
async def on_stop():
    '''Stop handler to handle stop event'''
    await cl.Message("You have stopped the task!").send()

@cl.set_starters
async def set_starters():
    '''Customise Harry Potter chat starters!'''
    return [
        cl.Starter(
            label="Who is Severus Snape",
            message="Who is Severus Snape?",
            icon="/public/logo_light.png"
        ),

        cl.Starter(
            label="Who is Albus Dumbledore",
            message="Who is Albus Dumbledore?"
        ),

        cl.Starter(
            label="Who is Lord Voldemort",
            message="Who is Lord Voldemort and how did he come back to life?"
        ),
    ]

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Handler function to resume a chat"""
    
    ## Restore memory buffer
    memory = ChatMemoryBuffer.from_defaults()
    root_messages = [m for m in thread["steps"]]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.put(
                ChatMessage(
                    role=MessageRole.USER,
                    content=message['output']
                )
            )
        else:
            memory.put(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=message['output']
                )
            )
    # set memory for user session - good practice for async deployment
    cl.user_session.set("memory", memory)

    # define service context (with Callback handler)
    service_context = Settings.callback_manager

    # define chat engine
    chat_engine = index.as_chat_engine(chat_mode="condense_question", llm=Settings.llm, streaming=True, verbose=True, service_context=service_context)

    # Set chat_engine for user session
    cl.user_session.set("chat_engine", chat_engine)

    # Output user info
    user = cl.user_session.get("user")
    logger.info(f"{user} has resumed chat")

@cl.on_logout
def on_logout(request: Request, response: Response):
    ### Handler to tidy up resources
    for cookie_name in request.cookies.keys():
        response.delete_cookie(cookie_name)
