# https://python.langchain.com/docs/langserve#server
from fastapi import FastAPI
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langserve import add_routes

## May be useful later
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_community.document_transformers import LongContextReorder
from functools import partial
from operator import itemgetter

from langchain_community.vectorstores import FAISS

## TODO: Make sure to pick your LLM and do your prompt engineering as necessary for the final assessment
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
instruct_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

## Load the document store
try:
    import os
    if os.path.exists("docstore_index.tgz"):
        os.system("tar xzf docstore_index.tgz")
    docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
    print(f"Loaded docstore with {len(docstore.docstore._dict)} documents")
except Exception as e:
    print(f"Failed to load docstore: {e}")
    # Create empty docstore as fallback
    from faiss import IndexFlatL2
    from langchain_community.docstore.in_memory import InMemoryDocstore
    embed_dims = len(embedder.embed_query("test"))
    docstore = FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

## PRE-ASSESSMENT: Run as-is and see the basic chain in action
add_routes(
    app,
    instruct_llm,
    path="/basic_chat",
)

## ASSESSMENT TODO: Implement these components as appropriate

# Generator: takes input dict with 'input' and 'context', returns string response
chat_prompt = ChatPromptTemplate.from_template(
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked you a question: {input}\n\n"
    " The following information may be useful for your response: "
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational)"
    "\n\nUser Question: {input}"
)

generator_chain = chat_prompt | instruct_llm | StrOutputParser()

add_routes(
    app,
    generator_chain,
    path="/generator",
)

# Retriever: takes string input, returns list of Document objects
retriever_chain = docstore.as_retriever()

add_routes(
    app,
    retriever_chain,
    path="/retriever",
)

## Might be encountered if this were for a standalone python file...
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9012)
