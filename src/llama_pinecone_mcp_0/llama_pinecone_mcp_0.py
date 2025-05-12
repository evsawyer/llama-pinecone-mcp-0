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
from typing import List, Any, Dict, Union
from enum import Enum
from pydantic import BaseModel, Field
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
import json

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

# Create an enum for the allowed operator strings
class Operator(str, Enum):
    EQ = "=="           # equals
    GT = ">"            # greater than
    LT = "<"            # less than
    NE = "!="           # not equal
    GTE = ">="          # greater than or equal to
    LTE = "<="          # less than or equal to
    IN = "in"           # value in list
    NIN = "nin"         # value not in list
    ANY = "any"         # any value in list matches
    ALL = "all"         # all values in list match
    TEXT_MATCH = "text_match"              # text match
    TEXT_MATCH_INSENSITIVE = "text_match_insensitive"  # text match (case insensitive) 
    CONTAINS = "contains"  # contains substring
    IS_EMPTY = "is_empty"  # field is empty

# Pydantic model for a single filter
class FilterConfig(BaseModel):
    key: str = Field(..., description="The metadata field name to filter on")
    value: Any = Field(..., description="The value to compare against")
    operator: Operator = Field(
        default="==",
        description="The operator to use for comparison"
    )
    model_config = {"arbitrary_types_allowed": True}

# Pydantic model for all filters
class FiltersConfig(BaseModel):
    filters: List[FilterConfig] = Field(
        default_factory=list,
        description="List of filter configurations"
    )

# Mapping between our Operator enum and FilterOperator from the library
OPERATOR_MAPPING = {
    "==": FilterOperator.EQ,
    ">": FilterOperator.GT,
    "<": FilterOperator.LT,
    "!=": FilterOperator.NE,
    ">=": FilterOperator.GTE,
    "<=": FilterOperator.LTE,
    "in": FilterOperator.IN,
    "nin": FilterOperator.NIN,
    "any": FilterOperator.ANY,
    "all": FilterOperator.ALL,
    "text_match": FilterOperator.TEXT_MATCH,
    "text_match_insensitive": FilterOperator.TEXT_MATCH_INSENSITIVE,
    "contains": FilterOperator.CONTAINS,
    "is_empty": FilterOperator.IS_EMPTY
}

@mcp.tool()
def create_metadata_filters(filters_config: FiltersConfig) -> MetadataFilters:
    """Create a MetadataFilters object based on a filter configuration."""
    
    # Get the filter configs from the FiltersConfig object
    filter_configs = filters_config.filters
    
    # Create the MetadataFilter objects
    filters = [
        MetadataFilter(
            key=config.key, 
            value=config.value, 
            operator=OPERATOR_MAPPING[config.operator.value]
        )
        for config in filter_configs
    ]
    return MetadataFilters(filters=filters)

@mcp.tool()
def parse_document(url: list[str], metadata: Schema) -> str:
    """Parse a list of URLs and return the parsed result."""
    documents = parser.load_data(url)
    for document in documents:
        document.metadata = metadata.model_dump()
    return documents

# @mcp.tool()
# def parse_documents(urls: list[str]) -> str:
#     """Parse a list of URLs and return the parsed result."""
#     return parser.load_data(urls)

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
def query(query: str, filters_config: FiltersConfig) -> str:
    """Query the Pinecone index and return the results."""
    filters = create_metadata_filters(filters_config)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace='cloudinary')
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = index.as_query_engine(filters=filters)
    return query_engine.query(query)

@mcp.tool()
def retrieve(query: str, filters_config: FiltersConfig, top_k: int) -> str:
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace='cloudinary')
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    filters = create_metadata_filters(filters_config)
    retriever = index.as_retriever(filters=filters, similarity_top_k=top_k)
    retrieved = retriever.retrieve(query)
    results = {}
    for i, vector in enumerate(retrieved):
        metadata = vector.metadata
        text = vector.text
        results[f"result {i+1}"] = {
            "metadata": metadata,
            "text": text
        }
    return results

