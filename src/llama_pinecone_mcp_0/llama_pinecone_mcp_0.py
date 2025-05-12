from fastmcp import FastMCP
import logging
import sys
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
from llama_parse import LlamaParse
from pydantic import BaseModel, Field

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

mcp = FastMCP("Pinecone")
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("quickstart")

parser = LlamaParse(
    api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),  # get API key from environment variables
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True,
)

# vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace='cloudinary')
# index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
# query_engine = index.as_query_engine()
 # validate Metadata
class Schema(BaseModel):
    source: str = Field(description = 'where this document came from')
    user_id: str = Field(description = 'ivc email of uploader')
    client: str = Field(description = 'client this document is for')
    title: str = Field(description = 'the title of the document')
    tag: list = Field(description = 'list of tags of the document e.g. ["cannabis"]')
            
    class Config:
        extra = "allow"

@mcp.tool()
def parse_document(url: list[str], metadata: Schema) -> str:
    """Parse a list of URLs and return the parsed result."""
    documents = parser.load_data(url)
    for document in documents:
        document.metadata = metadata
    return documents

@mcp.tool()
def parse_documents(urls: list[str]) -> str:
    """Parse a list of URLs and return the parsed result."""
    return parser.load_data(urls)

@mcp.tool()
def upsert_documents(url: list[str], metadata: Schema) -> str:
    """Upsert a document into the Pinecone index."""
    documents = parse_document(url, metadata)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace='cloudinary')
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        documents, storage_context=storage_context)
    return "Documents upserted successfully"

@mcp.tool()
def query_pinecone(query: str) -> str:
    """Query the Pinecone index and return the results."""
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace='cloudinary')
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = index.as_query_engine()
    return query_engine.query(query)
