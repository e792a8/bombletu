import asyncio
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from config import *
from adapt import GiteeAIEmbeddings, ChatMux
from langfuse import get_client

logger = get_log(__name__)


llm = ChatOpenAI(
    rate_limiter=InMemoryRateLimiter(requests_per_second=1),
    max_retries=30,
    temperature=0.6,
    model=ENV["LLM_MODEL"],
    api_key=ENV["LLM_API_KEY"],  # type:ignore
    base_url=ENV["LLM_BASE_URL"],
)

embed = GiteeAIEmbeddings(
    dimensions=int(ENV["EMBED_DIMENSIONS"]),
    model=ENV["EMBED_MODEL"],
    base_url=ENV["EMBED_BASE_URL"],
    api_key=ENV["EMBED_API_KEY"],  # type: ignore
)

langfuse = get_client()


def make_chroma(col: str, persist_dir: str | None = None):
    # embed = OpenAIEmbeddings(
    return Chroma(col, embedding_function=embed, persist_directory=persist_dir)
    # from chromadb import Documents, Embeddings, EmbeddingFunction
    # import numpy as np

    # class EmbedWrap(EmbeddingFunction):
    #     def __init__(self, *args, **kwargs):
    #         super().__init__(*args, **kwargs)

    #     @staticmethod
    #     def name():
    #         return "embed_wrap"

    #     def __call__(self, input: Documents) -> Embeddings:
    #         return [np.array(em) for em in embed.embed_documents(input)]

    # chroma = PersistentClient(persist_dir)
    # coll = chroma.get_or_create_collection(col, embedding_function=EmbedWrap())  # type: ignore
    # return coll


"""
def make_mem0() -> AsyncMemory:
    mem0_config = {
        "vector_store": {
            "provider": "langchain",
            "config": {"client": make_chroma("mem0", DATADIR + "/chroma")},
        },
        "llm": {"provider": "langchain", "config": {"model": llm}},
        "embedder": {"provider": "langchain", "config": {"model": embed}},
        "reranker": {
            "provider": "llm_reranker",
            "config": {
                "llm": {"provider": "langchain", "config": {"model": llm}},
            },
        },
    }
    mem = asyncio.run(AsyncMemory.from_config(mem0_config))
    return mem
"""
