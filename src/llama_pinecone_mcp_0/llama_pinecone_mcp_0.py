from fastmcp import FastMCP
import logging
import sys
import os

# needed for postgres engine
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index_cloud_sql_pg import PostgresEngine
from llama_index_cloud_sql_pg import PostgresDocumentStore

from llama_index.core.schema import NodeRelationship
from llama_parse import LlamaParse
from typing import List, Any, Optional
import uuid
from enum import Enum
from pydantic import BaseModel, Field
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
import asyncio
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

mcp = FastMCP("Pinecone")
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("quickstart")

# laod the index
vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace='mcp-test-2025-05-14')

engine = asyncio.run(PostgresEngine.afrom_instance(
    project_id="knowledge-base-458316",
    region="us-central1",
    instance="llamaindex-docstore",
    database="docstore",
    user="postgres",
    password="llamadocpass",
    ip_type="public",
))

doc_store = asyncio.run(PostgresDocumentStore.create(
    engine=engine,
    table_name="document_store",
))

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
    ],
    docstore=doc_store,
    vector_store=vector_store,
)

index = VectorStoreIndex.from_vector_store(pipeline.vector_store)

parser = LlamaParse(
    api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),  # get API key from environment variables
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True,
    split_by_page=False,
)

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
    operator: Operator = Field(..., description="The operator to use for comparison")
    model_config = {"arbitrary_types_allowed": True}

# Pydantic model for all filters
class FiltersConfig(BaseModel):
    filters: List[FilterConfig] = Field(
        default_factory=list,
        description="List of filter configurations"
    )

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
            operator=config.operator.value
        )
        for config in filter_configs
    ]
    return MetadataFilters(filters=filters)

@mcp.tool()
def parse_document(url: str, metadata: Schema, id: str) -> str:
    """Parse a list of URLs and return the parsed result."""
    document = parser.load_data(url)
    doc_id = id if id is not None else str(uuid.uuid4())
    for doc in document:
        doc.metadata = metadata.model_dump()
        doc.id_ = doc_id
    return document


@mcp.tool()
def insert_document(url: str, metadata: Schema, id: str) -> str:
    """Upsert a document into the Pinecone index."""
    document = parse_document(url, metadata, id)
    nodes = pipeline.run(documents=document)
    index.insert_nodes(nodes)
    return f"Ingested {len(nodes)} Nodes"

@mcp.tool()
def query(query: str, filters_config: FiltersConfig) -> str:
    """Query the Pinecone index and return the results."""
    filters = create_metadata_filters(filters_config)
    query_engine = index.as_query_engine(filters=filters)
    return query_engine.query(query)

@mcp.tool()
def retrieve(query: str, filters_config: FiltersConfig, top_k: int) -> dict:
    filters = create_metadata_filters(filters_config)
    retriever = index.as_retriever(filters=filters, similarity_top_k=top_k)
    retrieved = retriever.retrieve(query)
    results = {}
    for i, vector in enumerate(retrieved):
        id = vector.id_
        score = vector.score
        metadata = vector.metadata
        text = vector.text
        parent_doc_id = vector.node.relationships.get(NodeRelationship.SOURCE)
        results[f"result {i+1}"] = {
            "id": id,
            "score": score,
            "metadata": metadata,
            "text": text,
            "parent_doc_id": parent_doc_id
        }
    return results

@mcp.tool()
def get_document(id: str) -> dict:
    document = doc_store.get_document(doc_id = id)
    results = {
        "id": document.id_,
        "metadata": document.metadata,
        "text": document.text
    }
    return results

@mcp.tool()
def delete_document(id: str) -> str:
    doc_store.delete_document(doc_id = id)
    index.delete_ref_doc(ref_doc_id = id)
    return f"Deleted document {id}"
