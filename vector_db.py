from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from openai import OpenAI
import numpy as np
import asyncio
import time
from app.utils.chunking import chunk_text, hash_content, preprocess_text
from app.utils.logger import LoggerSetup
from app.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX,
)
from pinecone.exceptions import PineconeApiException

import asyncio

logger = LoggerSetup.setup(__name__)
openai = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

index = None
NAMESPACE = "preamble"


# ---------------------------------------------------------------------
# In-memory fallback vector store
# ---------------------------------------------------------------------
class InMemoryVectorStore:
    """Local vector DB with cosine similarity, used as Pinecone fallback"""

    def __init__(self, namespace: str = "preamble"):
        self.namespace = namespace
        self.vectors = []

    def upsert(self, vectors, namespace=None):
        self.vectors.extend(vectors)
        logger.info(
            f"📦 In-memory store: {len(vectors)} vectors added (total={len(self.vectors)})"
        )

    def query(
        self, vector, top_k=2, include_metadata=True, namespace=None, filter=None
    ):
        def cosine_sim(v1, v2):
            v1 = np.array(v1)
            v2 = np.array(v2)
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            return float(np.dot(v1, v2) / denom) if denom else 0.0

        matches = []
        for v in self.vectors:
            if filter and "doc_hash" in filter:
                if v["metadata"].get("doc_hash") != filter["doc_hash"].get("$eq"):
                    continue
            score = cosine_sim(vector, v["values"])
            logger.info(f"{v}")
            matches.append(
                {
                    "id": v["id"],
                    "score": score,
                    "metadata": v["metadata"] if include_metadata else None,
                }
            )
        matches.sort(key=lambda x: x["score"], reverse=True)
        top_matches = matches[:top_k]
        top_score = top_matches[0]["score"] if top_matches else 0
        logger.info(
            f"🔍 In-memory store: {len(top_matches)} matches top score={top_score:.4f}"
        )
        return {"matches": top_matches}


# ---------------------------------------------------------------------
# Vector DB initialization
# ---------------------------------------------------------------------
async def init_vector_db():
    """Initialize Pinecone index or fallback to in-memory store"""
    global index
    loop = asyncio.get_running_loop()

    def _init():
        global index
        try:
            existing_indexes = pc.list_indexes()
            if PINECONE_INDEX not in existing_indexes:
                pc.create_index(
                    name=PINECONE_INDEX,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws", region=PINECONE_ENVIRONMENT or "us-east-1"
                    ),
                    deletion_protection="disabled",
                )
                logger.info(f"✅ Created new Pinecone index '{PINECONE_INDEX}'.")
            else:
                logger.info(f"✅ Index '{PINECONE_INDEX}' already exists.")

            index = pc.Index(PINECONE_INDEX)
            index.describe_index_stats()
            logger.info(f"✅ Connected to Pinecone index '{PINECONE_INDEX}'.")
        except PineconeApiException as e:
            if e.status == 409:
                logger.warning(
                    f"⚠️ Index '{PINECONE_INDEX}' already exists (409). Connecting instead."
                )
                index = pc.Index(PINECONE_INDEX)
            else:
                logger.error(f"❌ Pinecone init failed: {e}")
                raise
        except Exception as e:
            logger.warning(
                f"⚠️ Pinecone unavailable ({e}). Falling back to in-memory store."
            )
            index = InMemoryVectorStore(NAMESPACE)

    await loop.run_in_executor(None, _init)


# ---------------------------------------------------------------------
# Ingest facts into vector DB
# ---------------------------------------------------------------------
async def ingest_facts(text: str):
    """Break document into chunks, embed, and upsert to vector DB sequentially"""
    global index
    if not index:
        await init_vector_db()

    try:
        text = preprocess_text(text)
        doc_hash = hash_content(text)
        chunks = chunk_text(text, 3, 1)
        logger.info(f"🔄 Chunked document into {len(chunks)} segments.")

        vectors = []

        for i, chunk in enumerate(chunks):
            # Embed chunk sequentially
            resp = await asyncio.to_thread(
                openai.embeddings.create,
                model="text-embedding-ada-002",
                input=chunk["text"],
            )

            embedding = resp.data[0].embedding

            vectors.append(
                {
                    "id": f"{doc_hash}-{i}",
                    "values": embedding,
                    "metadata": {
                        **chunk["metadata"],
                        "text": chunk["text"],
                        "doc_hash": doc_hash,
                    },
                }
            )

        if hasattr(index, "upsert"):
            index.upsert(vectors, namespace=NAMESPACE)
        else:
            logger.error("Index has no upsert() method.")

        logger.info(f"📤 Ingested {len(vectors)} chunks successfully.")
        return len(vectors)
    except Exception as e:
        logger.error(f"❌ Document ingestion failed: {e}")
        raise


# ---------------------------------------------------------------------
# Query vector DB for relevant facts
# ---------------------------------------------------------------------
async def query_facts(query: str, top_k=2, doc_hash: str = None):
    """Query vector DB for top-K relevant fact chunks"""
    global index
    if not index:
        await init_vector_db()

    try:
        resp = await asyncio.to_thread(
            openai.embeddings.create,
            model="text-embedding-ada-002",
            input=query,
        )
        query_vector = resp.data[0].embedding
        filter = {"doc_hash": {"$eq": doc_hash}} if doc_hash else None

        if hasattr(index, "query"):
            result = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=NAMESPACE,
                filter=filter,
            )
            matches = getattr(result, "matches", result.get("matches", []))
        else:
            matches = []

        logger.info(f"🔍 Retrieved {len(matches)} relevant chunks.,")
        return matches
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        raise
